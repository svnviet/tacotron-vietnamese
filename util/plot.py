import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def plot_alignment(alignment, path, info=None):
	fig, ax = plt.subplots()
	im = ax.imshow(
		alignment,
		aspect = 'auto',
		origin = 'lower',
		interpolation = 'none')
	fig.colorbar(im, ax = ax)
	xlabel = 'Decoder timestep'
	if info is not None:
		xlabel += '\n\n' + info
	plt.xlabel(xlabel)
	plt.ylabel('Encoder timestep')
	plt.tight_layout()
	plt.savefig(path, format = 'png')
