import re



def generate_structure_name(structure):
    # Rule 1: Limit to 16 characters
    structure = structure[:16].capitalize()

    # Rule 2: Ensure unique values
    structure = structure.lower()

    # Rule 3: Identify compound structures
    if structure[-1] in ['s', 'i']:
        structure += 'es'

    # Rule 4: Capitalize first character of each structure category, CamelCase
    structure = '_'.join([word.capitalize() for word in structure.split('_')])

    # Rule 5: Remove spaces
    structure = structure.replace(' ', '')

    # Rule 6: Add spatial categorizations
    structure = re.sub(r'_(L|R|A|P|I|S|RUL|RLL|RML|LUL|LLL|NAdj|Dist|Prox)$', r'\1', structure)

    # Rule 7: Ensure a consistent root structure name
    structure = re.sub(r'_\w+_', '_', structure)

    # Rule 8: Use standard category roots
    standard_roots = ['A', 'V', 'LN', 'CN', 'Glnd', 'Bone', 'Musc', 'Spc', 'VB', 'Sinus']
    for root in standard_roots:
        if structure.startswith(root):
            break
    else:
        # No standard root found, add a default
        structure = 'A_' + structure

    # Rule 9: Add PRV indication
    if 'PRV' in structure:
        structure = re.sub(r'_(PRV\d+)$', r'\1', structure)

    # Rule 10: Add partial structure designation
    if '_' in structure:
        root, _ = structure.split('_', 1)
        structure += f'_{root}'

    return structure