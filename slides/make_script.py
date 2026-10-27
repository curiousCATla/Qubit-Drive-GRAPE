#!/usr/bin/env python3
"""
Generate est_theory_script.md from content.tex.

The speaker script and the \note{} blocks in the deck are the same text, so it
is written once (in content.tex) and this script renders the markdown copy.
Run it after editing any \note{}:

    python3 slides/make_script.py

One markdown block per frame, in frame order, with the heading taken from the
frame title -- so slide N in est_theory_slides.pdf is section N here.
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# Custom macros from preamble.tex, rendered as readable plain text rather than
# LaTeX.  Longest first, so \ad is not eaten by a prefix of \aop etc.
MACROS = [
    (r'\brket{', 2, lambda a, b: f'<{a}|{b}>'),
    (r'\melem{', 3, lambda a, b, c: f'<{a}|{b}|{c}>'),
    (r'\ket{', 1, lambda a: f'|{a}>'),
    (r'\bra{', 1, lambda a: f'<{a}|'),
    (r'\proj{', 1, lambda a: f'|{a}><{a}|'),
    (r'\PE{', 1, lambda a: f'P_(E_{a})'),
    # Brace-matched, so nested arguments such as \sqrt{\lambda_{\max}(X)} work.
    (r'\sqrt{', 1, lambda a: f'sqrt({a})'),
    (r'\mathrm{', 1, lambda a: a),
    (r'\mathcal{', 1, lambda a: a),
    (r'\tfrac{', 2, lambda a, b: f'({a}/{b})'),
    (r'\frac{', 2, lambda a, b: f'({a}/{b})'),
]

# Applied in order; \dagger and the brace escapes are handled in textify().
SIMPLE = [
    (r'\dqec', 'Delta_QEC'), (r'\Lleak', 'L_(E_j)'), (r'\etam', 'eta_(E_j,psi)'),
    (r'\PC', 'P_C'), (r'\Cs', 'C'), (r'\Es', 'E'),
    (r'\aop', 'a'), (r'\ad', 'a^dag'), (r'\qop', 'q'), (r'\qd', 'q^dag'),
    (r'\nop', 'n'), (r'\id', 'I'), (r'\Tr', 'Tr'), (r'\spn', 'span'),
    (r'\bar n', 'nbar'), (r'\sum', 'sum'), (r'\cdots', '...'), (r'\cdot', '.'), (r'\pm', '±'),
    (r'\Delta', 'Delta'), (r'\partial', 'd'), (r'\Omega', 'Omega'), (r'\omega', 'omega'),
    (r'\max', 'max'), (r'\min', 'min'), (r'\exp', 'exp'), (r'\log', 'log'),
    (r'\alpha', 'alpha'), (r'\chi', 'chi'), (r'\delta', 'delta'),
    (r'\varepsilon', 'varepsilon'), (r'\epsilon', 'epsilon'), (r'\eta', 'eta'), (r'\lambda', 'lambda'),
    (r'\mu', 'mu'), (r'\phi', 'phi'), (r'\psi', 'psi'), (r'\rho', 'rho'),
    (r'\sigma', 'sigma'), (r'\tau', 'tau'), (r'\pi', 'pi'),
    (r'\propto', ' ∝ '), (r'\approx', ' ≈ '), (r'\times', 'x'), (r'\sim', '~'),
    (r'\Longrightarrow', ' => '), (r'\Rightarrow', ' => '),
    (r'\rightarrow', ' -> '), (r'\to', ' -> '),
    (r'\forall', 'for all '), (r'\in', ' in '), (r'\leq', ' <= '), (r'\neq', ' != '),
    (r'\vec', ''), (r'\tilde', ''), (r'\bar', 'conj '), (r'\hat', ''), (r'\,', ' '), (r'\;', ' '),
    (r'\!', ''), (r'\ ', ' '), (r'\%', '%'), (r'\&', '&'),
]


def expand(s):
    """Expand the deck's macros into plain text."""
    for name, nargs, render in MACROS:
        while name in s:
            i = s.index(name)
            j = i + len(name) - 1          # at the opening brace
            args = []
            for _ in range(nargs):
                depth, k = 0, j
                while k < len(s):
                    if s[k] == '{':
                        depth += 1
                    elif s[k] == '}':
                        depth -= 1
                        if depth == 0:
                            break
                    k += 1
                args.append(s[j + 1:k])
                j = k + 1
            s = s[:i] + render(*args) + s[j:]
    for name, rep in SIMPLE:
        s = s.replace(name, rep)
    return s


def textify(s):
    """LaTeX prose -> markdown prose."""
    # ~ is a LaTeX non-breaking space, and \dagger must collapse onto a
    # preceding ^ so that "E^{\dagger}" does not come out as "E^^dag".
    s = s.replace('~', ' ')
    s = s.replace(r'\|', '||')
    # \tfrac12 / \frac12 -- the brace-free two-digit shorthand
    s = re.sub(r'\\t?frac(\d)(\d)', r'(\1/\2)', s)
    s = re.sub(r'\^\{?\\dagger\}?', '^dag', s)
    s = s.replace('\\dagger', 'dag')
    s = expand(s)
    s = re.sub(r'\\textbf\{([^{}]*)\}', r'**\1**', s)
    s = re.sub(r'\\emph\{([^{}]*)\}', r'*\1*', s)
    s = re.sub(r'\\texttt\{([^{}]*)\}', r'`\1`', s)
    s = re.sub(r'\\text\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\tfrac\{([^{}]*)\}\{([^{}]*)\}', r'(\1/\2)', s)
    s = re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r'(\1/\2)', s)
    s = re.sub(r'\\sqrt\s*(\d)', r'sqrt(\1)', s)
    s = re.sub(r'\\(langle|rangle|left|right|quad|qquad|medskip|vfill|par|normalfont|small)\b ?', '', s)
    s = s.replace('---', '\u2014').replace('--', '\u2013')
    s = s.replace('``', '"').replace("''", '"')
    s = s.replace('\\{', '{').replace('\\}', '}').replace('$', '')
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def brace_body(s, i):
    """Given index of an opening '{', return (body, index after matching '}')."""
    depth, k = 0, i
    while k < len(s):
        if s[k] == '{':
            depth += 1
        elif s[k] == '}':
            depth -= 1
            if depth == 0:
                return s[i + 1:k], k + 1
        k += 1
    raise ValueError('unbalanced braces')


def main():
    src = open(os.path.join(HERE, 'content.tex')).read()

    # Frames after \\appendix are backup slides; the deck numbers them B1, B2, ...
    # (appendixnumberbeamer), so the script does too.
    appendix_at = src.find('\n\\appendix')
    frames = []
    for m in re.finditer(r'\\begin\{frame\}(\[[^\]]*\])?', src):
        end = src.index(r'\end{frame}', m.end())
        body = src[m.end():end]
        # title: a brace group immediately after \begin{frame}[opts]
        title = None
        if body.lstrip().startswith('{'):
            title, _ = brace_body(body, body.index('{'))
        # note block
        note_items = []
        ni = body.find(r'\note{')
        if ni >= 0:
            note, _ = brace_body(body, body.index('{', ni))
            inner = re.search(r'\\begin\{ntitem\}(.*)\\end\{ntitem\}', note, re.S)
            if inner:
                raw = inner.group(1)
                note_items = [textify(x) for x in raw.split(r'\item') if x.strip()]
        backup = appendix_at >= 0 and m.start() > appendix_at
        frames.append((title, note_items, backup))

    out = [
        '# Speaker script \u2014 EsT theory and pulse optimisation',
        '',
        'Generated from `content.tex` by `make_script.py`; edit the `\\note{}` blocks',
        'there, not this file. One section per slide, numbered to match',
        '`est_theory_slides.pdf`.',
        '',
        'Paper: Roy, Wetherbee & Fatemi, *Error Semitransparent Universal Control of a',
        'Bosonic Logical Qubit*, arXiv:2603.15356. Equation numbers are the paper\'s.',
        '',
        '---',
        '',
    ]
    n_main = n_backup = 0
    for title, items, backup in frames:
        head = textify(title) if title else 'Title slide'
        if backup:
            n_backup += 1
            label = f'B{n_backup}'
        else:
            n_main += 1
            label = str(n_main)
        out.append(f'## {label}. {head}')
        out.append('')
        if items:
            for it in items:
                out.append(f'- {it}')
        else:
            out.append('*(no notes)*')
        out.append('')

    text = '\n'.join(out)
    # \note appears verbatim in the header above, by design.
    leftover = sorted(set(re.findall(r'\\[A-Za-z]+', text)) - {r'\note'})
    if leftover:
        print('WARNING unconverted macros:', ' '.join(leftover))

    path = os.path.join(HERE, 'est_theory_script.md')
    open(path, 'w').write(text)
    print(f'wrote {path}: {n_main} slides + {n_backup} backup, '
          f'{sum(len(i) for _, i, _ in frames)} bullets')


if __name__ == '__main__':
    main()
