"""Summarise a unittest output file: each FAIL/ERROR id with its final exception line."""
import re, sys
for path in sys.argv[1:]:
    text = open(path).read()
    blocks = re.split(r'\n={70}\n', text)
    print(f"=== {path.split('/')[-1]}")
    for b in blocks[1:]:
        lines = b.split('\n')
        head = lines[0]
        if not re.match(r'(FAIL|ERROR):', head):
            continue
        body = b.split('-' * 70 + '\n', 1)[1] if ('-' * 70) in b else ''
        exc = [l for l in body.split('\n') if re.match(r'^[A-Za-z_.]*(Error|Exception|Mismatch|Write|Miss|Divergence|Warning)\b', l) or l.startswith('AssertionError')]
        last = exc[-1] if exc else body.strip().split('\n')[-1]
        print(f"{head}\n    -> {last[:300]}")
    m = re.findall(r'^(Ran .*|OK.*|FAILED.*)$', text, re.M)
    print('   ', m)
