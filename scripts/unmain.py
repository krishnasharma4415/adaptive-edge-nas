import os, glob
for py_file in glob.glob('scripts/*.py'):
    with open(py_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    in_main = False
    altered = False
    
    for line in lines:
        if line.startswith('if __name__ == "__main__":') or line.startswith("if __name__ == '__main__':"):
            in_main = True
            altered = True
            continue
        
        if in_main:
            if line.startswith('    '):
                new_lines.append(line[4:])
            elif line.startswith('\t'):
                new_lines.append(line[1:])
            elif line.strip() == '':
                new_lines.append(line)
            else:
                in_main = False
                new_lines.append(line)
        else:
            new_lines.append(line)
            
    if altered:
        with open(py_file, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f'Modified {py_file}')
