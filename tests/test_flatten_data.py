from deepdisc.data_format.file_io import get_data_from_json
from deepdisc.data_format.flatten import flatten_dc2
import numpy as np

ddicts = get_data_from_json('./deepdisc/test_data/dc2/single_test.json')

flatdat = flatten_dc2(ddicts)

print(flatdat)

np.save('flatdat_test.npy', flatdat)



