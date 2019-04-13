import matplotlib.pyplot as plt

def write_img_pair(img1, img2, text1, text2, dst):
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
	f.tight_layout()
	ax1.imshow(img1)
	ax1.set_title(text1, fontsize=20)
	ax2.imshow(img2)
	ax2.set_title(text2, fontsize=20)
	plt.subplots_adjust(left=0., right=1, top=0.95, bottom=0.)
	plt.savefig(dst)
	plt.show()