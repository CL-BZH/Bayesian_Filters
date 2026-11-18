"""Generate the TikZ Bayesian-network panels (Chapter 1 and Appendix C) into figures/tikz/."""
DX, ZY, UY = 1.65, -1.25, 1.05
def pos(kind, k):
    if kind == 'x': return (DX*k, 0)
    if kind == 'z': return (DX*k, ZY)
    if kind == 'u': return (DX*k + DX/2, UY)
def lab(kind, k, names):
    return names[k]
def net(names, given=(), query=(), red=(), green=(), zobs=None, kmax=3,
        dots=True, ulist=None, scale=1.25, pre=(), extra=(), drop=()):
    """names: index labels for x_k, k=0..kmax. given/query: set of (kind,k).
    red/green: list of paths, each a list of (kind,k). zobs: set of k whose z
    is drawn (default 1..kmax or all)."""
    L = []
    L.append(r"\begin{tikzpicture}[scale=%s,>=Stealth," % scale)
    L.append(r"  var/.style={circle,draw,thick,minimum size=9.5mm,inner sep=0pt,fill=white,font=\small},")
    L.append(r"  ctl/.style={rectangle,draw,minimum size=6.5mm,inner sep=1.5pt,fill=white,font=\small},")
    L.append(r"  giv/.style={fill=defblue!22},")
    L.append(r"  qry/.style={draw=orange!85!black,line width=1.6pt},")
    L.append(r"  drp/.style={draw=violet!80!black,line width=1.6pt}]")
    for path, col in [(p, 'red!70') for p in red] + [(p, 'green!60!black') for p in green]:
        pts = " -- ".join("(%.3f,%.3f)" % pos(*n) for n in path)
        L.append(r"  \draw[line width=5pt,%s,opacity=0.35,line cap=round,line join=round] %s;" % (col, pts))
    L.extend(pre)   # raw TikZ drawn under the nodes (curved bands)
    zs = range(kmax+1) if zobs is None else zobs
    us = range(kmax) if ulist is None else ulist
    def style(n, base):
        s = [base]
        if n in given: s.append('giv')
        if n in query: s.append('qry')
        if n in drop: s.append('drp')
        return ",".join(s)
    for k in range(kmax+1):
        x, y = pos('x', k)
        L.append(r"  \node[%s] (x%d) at (%.3f,%.3f) {$\x_{%s}$};" % (style(('x',k),'var'), k, x, y, names[k]))
    for k in zs:
        x, y = pos('z', k)
        L.append(r"  \node[%s] (z%d) at (%.3f,%.3f) {$\z_{%s}$};" % (style(('z',k),'var'), k, x, y, names[k]))
    for k in us:
        x, y = pos('u', k)
        L.append(r"  \node[%s] (u%d) at (%.3f,%.3f) {$\uu_{%s}$};" % (style(('u',k),'ctl'), k, x, y, names[k]))
    for k in range(kmax):
        L.append(r"  \draw[->,thick] (x%d) -- (x%d);" % (k, k+1))
    for k in us:
        L.append(r"  \draw[->,thick] (u%d) -- (x%d);" % (k, k+1))
    for k in zs:
        L.append(r"  \draw[->,thick] (x%d) -- (z%d);" % (k, k))
    if dots:
        L.append(r"  \draw[->,thick,dotted] ($(x0)+(-1.1,0)$) -- (x0);")
        L.append(r"  \draw[->,thick,dotted] (x%d) -- ++(1.1,0);" % kmax)
    L.extend(extra)   # raw TikZ drawn last (extra arrows)
    L.append(r"\end{tikzpicture}")
    return "\n".join(L)

T = ['t-1', 't', 't+1', 't+2']
X = lambda k: ('x', k); Z = lambda k: ('z', k); U = lambda k: ('u', k)
panels = {
 # given = S (stays behind the bar), query = X, drop = Y (to remove)
 'A2': net(T, given={X(1)}, query={Z(1)}, drop={X(0), Z(0), U(0)},
           red=[[Z(0), X(0), X(1), Z(1)], [U(0), X(1), Z(1)]]),
 'A1': net(T, given={X(1), U(1)}, query={X(2)},
           drop={X(0), Z(0), Z(1), U(0)},
           red=[[Z(0), X(0), X(1), X(2)], [Z(1), X(1), X(2)],
                [U(0), X(1), X(2)]]),
 'pred': net(T, given={Z(0), Z(1), U(0)}, query={X(1)}, drop={U(1)},
           red=[[X(1), X(2), U(1)]]),
 'smooth': net(T, given={Z(0), Z(1), U(0), U(1), U(2)}, query={X(1)},
           drop={Z(2)}, green=[[X(1), X(2), Z(2)]]),
}
# Controls computed during the run (feedback): extra arrows in blue.
FB = r"\draw[->,thick,blue!70!black]"
BEND = "to[out=35,in=-100]"
panels['fbz'] = net(T, given={Z(0), Z(1), U(0)}, query={X(1)}, drop={U(1)},
    red=[[X(1), X(2), U(1)]],
    pre=[r"  \draw[line width=5pt,red!70,opacity=0.35,line cap=round,line join=round] "
         r"(%.3f,%.3f) -- (%.3f,%.3f) %s (%.3f,%.3f);" % (*pos('x', 1), *pos('z', 1), BEND, *pos('u', 1))],
    extra=[r"  %s (z%d) %s (u%d);" % (FB, k, BEND, k) for k in range(3)])
panels['fbx'] = net(T, given={Z(0), Z(1), U(0)}, query={X(1)}, drop={U(1)},
    red=[[X(1), X(2), U(1)]], green=[[X(1), U(1)]],
    extra=[r"  %s (x%d) -- (u%d);" % (FB, k, k) for k in range(3)])
main = net(['0', '1', '2', '3'], given={Z(1), Z(2), Z(3), U(0), U(1), U(2)},
           zobs=[1, 2, 3], dots=False, scale=1.3)
import os, sys
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "tikz")
for k, v in panels.items():
    open(f"{out}/bn_{k}.tex", "w").write(v + "\n")
open(f"{out}/bn_main.tex", "w").write(main + "\n")
