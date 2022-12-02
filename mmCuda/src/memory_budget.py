

# returns the memory needed to allocate a vector of given size and type
def get_memory_usage(num_entries, dtype='int32', human_readable=True):

	if dtype == 'int32':
		nbyte = 4
	elif dtype == 'int16':
		nbyte = 2

	B = nbyte * num_entries

	if human_readable:
		if B >= 1024:
			KB = B // 1024
			B  = B % 1024
		else:
			return f'{B}B'

		if KB >= 1024:
			MB = KB // 1024
			KB = KB % 1024
		else:
			return f'{KB}KB'

		if MB >= 1024:
			GB = MB // 1024
			MB = MB % 1024
		else:
			return f'{MB}MB {KB}KB'

		if GB >= 1024:
			TB = GB // 1024
			GB = GB % 1024
			return f'{TB}TB {GB}GB'
		else:
			return f'{GB}GB {MB}MB'

	return B



# module map usage

num_triplets = 1_242_265
num_doublets = 296_502

flatten_triplet_size   = num_triplets * 3
flatten_doublet_size   = num_doublets * 2
flatten_t2d_links_size = num_triplets * 2

# 6 triplet cuts and 6 doublet cuts (let's say)
cuts_size = num_triplets * 6 + num_doublets * 6

num_entries =  flatten_triplet_size + flatten_doublet_size + flatten_t2d_links_size + cuts_size

print(get_memory_usage(num_entries))



# event memory usage
num_entries = 2 * 200_000_000
print(get_memory_usage(num_entries, 'int16'))



# cap at 400 option (Impossible)
num_entries = 400 * 400 * 2 * num_doublets
print(get_memory_usage(num_entries))

