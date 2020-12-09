
from ovito.io import import_file
from ovito.modifiers import CoordinationAnalysisModifier
import numpy as np

path = './46nvt.xyz'

# Load a particle dataset, apply the modifier, and evaluate pipeline.
pipeline = import_file(path)
modifier = CoordinationAnalysisModifier(cutoff = 1.85, number_of_bins = 200)
pipeline.modifiers.append(modifier)
data = pipeline.compute()

# Print the computed g(r) function values.
#print(data.tables['coordination-rdf'].xy())

np.savetxt("coordination.txt",data.particles['Coordination'],fmt='%d')
a = np.loadtxt("coordination.txt")
print('CN:',np.mean(a))
from os import remove
remove("coordination.txt")
