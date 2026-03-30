files = ['main.py', 'requirements.txt', 'Dockerfile']
for f in files:
    content = open(f, 'rb').read()
    fixed = content.replace(b'\r\n', b'\n')
    open(f, 'wb').write(fixed)
    print(f'Fixed: {f}')