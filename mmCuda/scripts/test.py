import uproot

'''
Read a ROOT file with uproot that contains 60K events.
Reduce to 50K events and write to a new ROOT file.
'''
def main():
    # Read the ROOT file
    root_file = uproot.open('test.root')
    tree = root_file['tree']
    # Reduce the number of events to 50K
    tree = tree[:50000]
    # Write the new ROOT file
    tree.write('test_reduced.root') 