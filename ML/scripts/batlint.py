"""Static check for the failure modes that have actually cost hours here."""
import re, sys, os

def lint(path):
    raw = open(path,'rb').read()
    txt = raw.decode('utf-8','replace')
    issues = []
    bare = sum(1 for i,b in enumerate(raw) if b==10 and (i==0 or raw[i-1]!=13))
    if bare: issues.append(f"{bare} bare LF line endings (Windows needs CRLF)")

    depth = 0
    for n,l in enumerate(txt.replace('\r\n','\n').split('\n'), 1):
        s = l.strip(); low = s.lower()
        if not s or low.startswith('rem') or s.startswith('::'):
            continue
        nq = re.sub(r'"[^"]*"', '""', s)          # ignore quoted text

        opens  = nq.rstrip().endswith('(')
        closes = nq.startswith(')')

        if depth > 0:
            # count parens that are NOT escaped with ^ and NOT the block delimiters
            n_open  = len(re.findall(r'(?<!\^)\(', nq)) - (1 if opens  else 0)
            n_close = len(re.findall(r'(?<!\^)\)', nq)) - (1 if closes else 0)
            # a backticked `...` for /f command is allowed to contain parens
            if '`' in nq:
                n_open = n_close = 0
            if n_close > 0:
                issues.append(f"line {n}: unescaped ')' inside a block -> ENDS THE BLOCK EARLY: {s[:64]}")
            elif n_open > 0:
                issues.append(f"line {n}: unescaped '(' inside a block: {s[:64]}")

        if opens:   depth += 1
        elif closes: depth = max(0, depth-1)

    if depth: issues.append(f"unbalanced parentheses at EOF (depth {depth})")
    return issues

worst = 0
for f in sorted(sys.argv[1:]):
    iss = lint(f)
    print(f"[{'OK  ' if not iss else 'FAIL'}] {os.path.basename(f)}")
    for i in iss: print(f"        - {i}")
    worst = max(worst, len(iss))
sys.exit(1 if worst else 0)
