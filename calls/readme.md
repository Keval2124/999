├── calls
    └── index.py

data cleaning:

# 1-liner in bash (or do it in Python if you prefer)
sed -i \
    -e 's/OPERATOR:/\nOPERATOR:/g' \
    -e 's/CALLER:/\nCALLER:/g' \
    -e 'y/ABCDEFGHIJKLMNOPQRSTUVWXYZ/abcdefghijklmnopqrstuvwxyz/' \
    -e 's/^[[:space:]]*//;s/[[:space:]]*$//' \
    -e '/^$/d' \
    -e 's/911/999/g' \
    -e 's/9-1-1/9-9-9/g' \
    -e 's/^operator:[[:space:]]*/operator: /' \
    -e 's/^caller:[[:space:]]*/caller: /' \
    -e '/^\(operator\|caller\):/!s/^/caller: /' \
    additional_911_dialogs.txt