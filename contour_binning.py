import os
import sys
import math
import numpy as np
from astropy.io import fits

class ContourBin:
	"""Class for performing contour binning on astronomical images."""

	def __init__(
		self,
		fitsfile,
		sn_ratio,
		smooth,
		constrain_val,
		reg_bin,
		automask=False,
		maskname=None,
		src_exposure_filename=None,
		bkg_filename=None,
		bkg_exposure_filename=None,
		noisemap_filename=None,
		smoothed_filename=None,
		psf_filename=None
	):
		"""Initialize ContourBin instance."""
		self.filename = fitsfile
		self.sn_ratio = sn_ratio
		self.smooth = smooth
		self.constrain_val = constrain_val
		self.reg_bin = reg_bin
		self.automask = automask
		self.maskname = maskname
		self.src_exposure_filename = src_exposure_filename
		self.bkg_filename = bkg_filename
		self.bkg_exposure_filename = bkg_exposure_filename
		self.noisemap_filename = noisemap_filename
		self.smoothed_filename = smoothed_filename
		self.psf_filename = psf_filename

		self.load_all_data()

		# Load or estimate the smoothed data
		if self.smoothed_filename:
			try:
				print(f"Loading smoothed image {self.smoothed_filename}")
				self.smoothed_data, _ = self.load_image(self.smoothed_filename, verbose=False)
			except FileNotFoundError:
				print(f"Smoothed image {self.smoothed_filename} not found. Estimating flux instead.")
				self.smoothed_data = self.estimate_flux(
					self.source_data,
					self.bkg_data,
					self.mask,
					self.exposuremap,
					self.bkgexpmap,
					self.noisemap,
					self.smooth
				)
		else:
			print(f"\nSmoothing data (S/N = {self.smooth})")
			self.smoothed_data = self.estimate_flux(
				self.source_data,
				self.bkg_data,
				self.mask,
				self.exposuremap,
				self.bkgexpmap,
				self.noisemap,
				self.smooth
			)

		if self.smoothed_data.shape != self.source_data.shape:
			raise ValueError("Input image does not match smoothed image shape")

		# Save the smoothed data if not already provided
		output_dir = f"contour_binning_sn{self.sn_ratio}_smooth{self.smooth}_constrain{self.constrain_val}"
		if not self.smoothed_filename:
			os.makedirs(output_dir, exist_ok=True)
			hdu = fits.PrimaryHDU(self.smoothed_data)
			hdu.header = fits.getheader(self.filename)
			smoothed_output_path = os.path.join(output_dir, f"smoothed_data_smooth{self.smooth}.fits")
			hdu.writeto(smoothed_output_path, overwrite=True)
			print(f"Smoothed data image written to file {smoothed_output_path}")

		# Perform binning
		print(f"Performing binning with S/N threshold {self.sn_ratio}...")
		the_binner = Binner(
			in_image=self.source_data,
			smoothed_image=self.smoothed_data,
			threshold=self.sn_ratio
		)
		the_binner.set_back_image(self.bkg_data, self.exposuremap, self.bkgexpmap)
		the_binner.set_noisemap_image(self.noisemap)
		the_binner.set_mask_image(self.mask)
		the_binner.set_constrain_fill(self.constrain_val)
		the_binner.set_scrub_large_bins(self.reg_bin)

		print("\nStarting binning process...")
		the_binner.do_binning(True)  # bin_down=True
		print("Binning process completed...\n")

		# Call do_scrub after binning
		print("Starting scrubbing process...")
		the_binner.do_scrub()
		print("Scrubbing process completed...\n")

		output_image = the_binner.get_output_image()
		sn_image = the_binner.get_sn_image()
		binmap_image = the_binner.get_binmap_image()

		# Save the output images
		print("Creating output images...")
		os.makedirs(output_dir, exist_ok=True)
		self.save_image(os.path.join(output_dir, "contbin_out.fits"), output_image)
		self.save_image(os.path.join(output_dir, "contbin_sn.fits"), sn_image)
		self.save_image(os.path.join(output_dir, "contbin_binmap.fits"), binmap_image)
		self.save_image(os.path.join(output_dir, "contbin_mask.fits"), self.mask)

		print(f"Calculating statistics...")
		the_binner.calc_outputs(output_dir)
		print(f"Output images saved! Check {output_dir} for results...")
		print(f"ContourBin process completed successfully!")

	def load_all_data(self):
		"""Load all required data for binning."""
		# self.source_data, self.source_exposuretime = self.load_image(self.filename)
		try:
			self.source_data, self.source_exposuretime = self.load_image(self.filename)
		except FileNotFoundError as e:
			print(f"Error: Source file not found: {e}")
			sys.exit(1)

		# Load or create the mask
		if self.maskname:
			print(f"Loading masking image {self.maskname}")
			mask_data, _ = self.load_image(self.maskname)
			self.mask = (mask_data > 0).astype(np.short)
		elif self.automask:
			self.mask = self.auto_mask()
		else:
			self.mask = np.ones_like(self.source_data, dtype=np.short)

		# Load exposure map
		if self.src_exposure_filename:
			print(f"Loading given exposure map {self.src_exposure_filename}")
			self.exposuremap, _ = self.load_image(self.src_exposure_filename)
		else:
			print("Using blank exposure map (exp = 1.0)")
			self.exposuremap = np.full_like(self.source_data, 1.0)

		# Load background image
		if self.bkg_filename:
			print(f"Loading given background image {self.bkg_filename}")
			self.bkg_data, self.bkg_exposuretime = self.load_image(self.bkg_filename)
		else:
			self.bkg_data = None # np.zeros_like(self.source_data)

		# Load background exposure map
		if self.bkg_exposure_filename:
			print(f"Using given background exposure map {self.bkg_exposure_filename}")
			self.bkgexpmap, _ = self.load_image(self.bkg_exposure_filename)
		else:
			self.bkgexpmap = np.full_like(self.source_data, 1.0)
			print("Using blank background exposure (exp = 1.0)")

		# Correct division by zeros
		self.bkgexpmap[self.bkgexpmap < 1e-7] = 1e-7
		self.exposuremap[self.exposuremap < 1e-7] = 1e-7

		# Load noise map
		if self.noisemap_filename:
			print(f"Loading noise map {self.noisemap_filename}")
			self.noisemap, _ = self.load_image(self.noisemap_filename)
			if not (self.source_data.shape == self.noisemap.shape):
				raise ValueError("Noise map must have the same dimensions as the source image")
		else:
			self.noisemap = None  # Set to None if not provided
		
		# Load PSF map
		if self.psf_filename:
			print(f"Loading PSF map {self.psf_filename}")
			self.psf_map, _ = self.load_image(self.psf_filename)
			if not (self.source_data.shape == self.psf_map.shape):
				raise ValueError("PSF map must have the same dimensions as the source image")
		else:
			self.psf_map = None

		if not (self.source_data.shape == self.mask.shape == self.exposuremap.shape):
			raise ValueError("Input images must have the same dimensions")

	def load_image(self, filename, verbose=True):
		"""Load an image from a FITS file."""
		if verbose:
			print(f"Loading image {filename}")
		with fits.open(filename) as hdulist:
			image_data = hdulist[0].data
			exposure = hdulist[0].header.get('EXPOSURE', 1.0)  # Default to 1 if EXPOSURE is not present
		return image_data, exposure

	def save_image(self, filename, image):
		"""Save an image to a FITS file."""
		hdu = fits.PrimaryHDU(data=image)
		header = fits.getheader(self.filename)
		hdu.header = header
		# Additional metadata
		history = [
			f"Contbin (Jeremy Sanders) adapted to Python (by JP Breuer).",
			f"This filename: {filename}",
			f"Input image: {self.filename}",
			f"Back image: {self.bkg_filename}",
			f"Mask image: {self.maskname}",
			f"Smoothed image: {self.smoothed_filename}",
			f"Expmap image: {self.src_exposure_filename}",
			f"Back expmap image: {self.bkg_exposure_filename}",
			f"Noise map image: {self.noisemap_filename}",
			f"SN threshold: {self.sn_ratio}",
			f"Smooth SN: {self.smooth}",
			f"Automask: {self.automask}",
			f"Constrain val: {self.constrain_val}"
		]
		for entry in history:
			hdu.header.add_history(entry)
		hdu.writeto(filename, overwrite=True)

	def auto_mask(self):
		"""Automatically generate a mask for the image."""
		print("Automasking... ", end='', flush=True)

		blocksize = 8
		yw, xw = self.source_data.shape
		mask = np.ones_like(self.source_data, dtype=np.short)

		for y in range(0, yw, blocksize):
			for x in range(0, xw, blocksize):
				block = self.source_data[y:y + blocksize, x:x + blocksize]
				sum_ = np.sum(block)
				if abs(sum_) < 1e-5:
					mask[y:y + blocksize, x:x + blocksize] = 0

		print("Done")
		return mask


	@staticmethod
	def error_sqd_est(c):
		"""Estimate the error squared on c counts.
		Uses formula from Gehrels 1986 ApJ, 303, 336) eqn 7.
		"""
		# value = c + 0.75
		# if value < 0:
		# 	print(f"WARNING: Negative value in error_sqd_est: c={c}")
		# 	value = 0.0
		# return (1.0 + math.sqrt(value)) ** 2
		return (1.0 + math.sqrt(c + 0.75)) ** 2

	def estimate_flux(
		self,
		in_image,
		back_image,
		mask_image,
		expmap_image,
		bg_expmap_image,
		noisemap_image,
		minsn=10
	):
		"""Estimate the flux of the image."""
		yw, xw = in_image.shape
		max_radius = int(np.hypot(yw, xw)) + 1
		iteration_image = np.zeros((yw, xw))
		estimated_errors = np.zeros((yw, xw))

		# Precompute annuli points
		annuli_points = [[] for _ in range(max_radius)]
		for dy in range(-max_radius, max_radius + 1):
			for dx in range(-max_radius, max_radius + 1):
				r = int(np.hypot(dy, dx))
				if r < max_radius:
					annuli_points[r].append((dx, dy))
					# annuli_points[r].append((dy, dx))

		min_sn_2 = minsn ** 2

		for y in range(yw):
			print(f"\rSmoothing: {y * 100. / yw:.1f}%", end='')

			for x in range(xw):
				if mask_image[y, x] < 1:
					continue

				fg_sum = bg_sum = bg_sum_weight = expratio_sum_2 = 0.0
				noise_2_total = 0.0
				count = 0

				radius = 0
				sn_2 = 0.0

				while radius < max_radius and sn_2 < min_sn_2:
					for dy, dx in annuli_points[radius]:
						xp = x + dx
						yp = y + dy
						if not (0 <= xp < xw and 0 <= yp < yw):
							continue
						if mask_image[yp, xp] < 1:
							continue

						in_signal = in_image[yp, xp]

						if back_image is not None:
							bg = back_image[yp, xp]
							expratio = expmap_image[yp, xp] / bg_expmap_image[yp, xp]
							bg_sum += bg
							bg_sum_weight += bg * expratio
							expratio_sum_2 += expratio ** 2

						if noisemap_image is not None:
							noise_2_total += noisemap_image[yp, xp] ** 2

						fg_sum += in_signal
						count += 1

					if count > 0:
						if noisemap_image is not None:
							noise_2 = noise_2_total
						else:
							noise_2 = self.error_sqd_est(fg_sum)
						if back_image is not None and count > 0:
							noise_2 += (expratio_sum_2 / count) * self.error_sqd_est(bg_sum)

						sn_2 = (fg_sum - bg_sum_weight) ** 2 / noise_2
					else:
						sn_2 = 0.0

					radius += 1

				if count > 0:
					iteration_image[y, x] = (fg_sum - bg_sum_weight) / count
					estimated_errors[y, x] = np.sqrt(noise_2)
				else:
					iteration_image[y, x] = 0
					estimated_errors[y, x] = 0

		print("\nSmoothing completed.\n")
		return iteration_image

class BinHelper:
	"""Helper class for binning operations."""
	def __init__(self, in_image, smoothed_image, bins_image, threshold, psf_map=None):
		self.in_image = in_image
		self.smoothed_image = smoothed_image
		self.bins_image = bins_image
		self.threshold = threshold
		self.psf_map = psf_map

		self.xw = in_image.shape[1]
		self.yw = in_image.shape[0]

		self.back_image = None
		self.expmap_image = None
		self.bg_expmap_image = None
		self.noisemap_image = None
		self.mask_image = np.ones((self.yw, self.xw), dtype=np.int16)

		self.max_annuli = self.unsigned_radius(self.xw, self.yw) + 1
		self.bin_counter = 0

		self.constrain_fill = False
		self.constrain_val = 4
		self.scrub_large_bins = -1

		self.annuli_points = []
		self.areas = []

		self.bin_no_neigh = 4
		self.bin_neigh_x = [-1, 0, 1, 0]
		self.bin_neigh_y = [0, -1, 0, 1]

		self.precalculate_annuli()
		self.precalculate_areas()

	@staticmethod
	def unsigned_radius(x, y):
		"""Calculate unsigned radius."""
		return int(math.sqrt(x * x + y * y))
	
	def set_back(self, back_image, expmap_image, bg_expmap_image):
		"""Set background images."""
		self.back_image = back_image
		self.expmap_image = expmap_image
		self.bg_expmap_image = bg_expmap_image

	def set_noisemap(self, noisemap_image):
		"""Set noise map image."""
		self.noisemap_image = noisemap_image

	def set_mask(self, mask_image):
		"""Set mask image."""
		self.mask_image = mask_image

	def set_psf_map(self, psf_map):
		"""Set the PSF map."""
		self.psf_map = psf_map

	def set_constrain_fill(self, constrain_val):
		"""Set constraint fill value."""
		self.constrain_fill = True
		self.constrain_val = constrain_val

	def set_scrub_large_bins(self, fraction):
		"""Set scrub large bins fraction."""
		self._scrub_large_bins = fraction

	def bin_counter_increment(self):
		"""Increment and return the bin counter."""
		self.bin_counter += 1
		return self.bin_counter - 1

	def no_bins(self):
		"""Get number of bins."""
		return self.bin_counter

	def get_radius_for_area(self, area):
		"""Get radius for a given area."""
		return np.searchsorted(self.areas, area)

	def precalculate_annuli(self):
		"""Precalculate annuli points."""
		self.annuli_points = [[] for _ in range(self.max_annuli)]
		for dy in range(-self.yw + 1, self.yw):
			for dx in range(-self.xw + 1, self.xw):
				r = self.unsigned_radius(dx, dy)
				if r < self.max_annuli:
					self.annuli_points[r].append((dx, dy))
					# self.annuli_points[r].append((dy, dx))

	def precalculate_areas(self):
		"""Precalculate areas."""
		self.areas = []
		total = 0
		for radius in range(self.max_annuli):
			area = len(self.annuli_points[radius])
			total += area
			self.areas.append(total)

class Bin:
	"""Class representing a bin in the binning process."""
	def __init__(self, helper):
		self.helper = helper
		# self._bin_no = self._helper.bin_counter()
		self.bin_no = self.helper.bin_counter_increment()
		self.aimval = -1
		self.fg_sum = 0.0
		self.bg_sum = 0.0
		self.bg_sum_weight = 0.0
		self.noisemap_2_sum = 0.0
		self.expratio_sum_2 = 0.0
		self.centroid_sum = np.array([0.0, 0.0])
		self.centroid_weight = 0.0
		self.count = 0
		self.all_points = []
		self.edge_points = []
		self.bin_no_neigh = self.helper.bin_no_neigh
		self.bin_neigh_x = self.helper.bin_neigh_x
		self.bin_neigh_y = self.helper.bin_neigh_y

	@staticmethod
	def square(d):
		return d * d
	
	@staticmethod
	def error_sqd_est(c):
		"""Estimate the error squared on c counts.
		Uses formula from Gehrels 1986 ApJ, 303, 336) eqn 7.
		"""
		# value = c + 0.75
		# if value < 0:
		# 	print(f"WARNING: Negative value in error_sqd_est: c={c}")
		# 	value = 0.0
		# return (1.0 + math.sqrt(value)) ** 2
		return (1.0 + math.sqrt(c + 0.75)) ** 2
	
	def drop_bin(self):
		"""Drop the current bin."""
		self.fg_sum = 0.0
		self.bg_sum = 0.0
		self.bg_sum_weight = 0.0
		self.noisemap_2_sum = 0.0
		self.expratio_sum_2 = 0.0
		self.centroid_sum = np.array([0.0, 0.0])
		self.centroid_weight = 0.0
		self.count = 0
		self.all_points.clear()
		self.edge_points.clear()

	def do_binning(self, x, y):
		"""Perform binning starting from a seed point."""
		self._aimval = self.helper.smoothed_image[y, x]
		self.add_point(x, y)

		sn_threshold_2 = self.helper.threshold * self.helper.threshold

		while self.sn_2() < sn_threshold_2:
			if not self.add_next_pixel():
				break

	def count(self):
		return self.count

	def signal(self):
		"""Calculate the signal."""
		return self.fg_sum - self.bg_sum_weight

	def noise_2(self):
		"""Calculate the noise squared."""
		if self.helper.noisemap_image is None:
			# Using background image
			n = self.error_sqd_est(self.fg_sum)
			if self.helper.back_image is not None:
				n += (self.expratio_sum_2 / self.count) * self.error_sqd_est(self.bg_sum)
			return n
		else:
			# Using noisemap
			return self.noisemap_2_sum

	def sn_2(self):
		"""Calculate the signal-to-noise squared."""
		csignal = self.signal()
		cnoise_2 = self.noise_2()
		if cnoise_2 < 1e-7:
			return 1e-7
		else:
			return csignal * csignal / cnoise_2

	def check_constraint(self, x, y):
		"""Check if adding a point satisfies the constraint."""
		c = self.centroid_sum / self.centroid_weight
		dx = c[0] - x
		dy = c[1] - y
		r2 = dx * dx + dy * dy

		circradius = self.helper.get_radius_for_area(self.count) + 1

		return (r2 / (circradius * circradius)) < self.square(self.helper.constrain_val)

	def get_all_points(self):
		return self.all_points

	def get_edge_points(self):
		return self.edge_points

	def bin_no(self):
		return self.bin_no

	def set_bin_no(self, num):
		self.bin_no = num

	def add_point(self, x, y):
		"""Add a point to the bin."""
		self.all_points.append((x, y))

		signal = self.helper.in_image[y, x]
		self.fg_sum += signal
		self.count += 1
		self.helper.bins_image[y, x] = self.bin_no

		if self.helper.back_image is not None:
			bs = self.helper.expmap_image[y, x] / self.helper.bg_expmap_image[y, x]
			back = self.helper.back_image[y, x]
			self.bg_sum += back
			self.bg_sum_weight += back * bs
			self.expratio_sum_2 += bs * bs

			signal -= back * bs

		if self.helper.noisemap_image is not None:
			self.noisemap_2_sum += self.square(self.helper.noisemap_image[y, x])

		# Update centroid
		cs = max(signal, 1e-7)
		self.centroid_sum += np.array([x, y]) * cs
		self.centroid_weight += cs

		# Put into edge (it might not be, but it will get flushed out)
		if (x, y) not in self.edge_points:
			self.edge_points.append((x, y))

	def remove_point(self, x, y):
		"""Remove a point from the bin."""
		P = (x, y)
		if P in self.all_points:
			self.all_points.remove(P)
		else:
			raise ValueError("Point not in _all_points")

		if P in self.edge_points:
			self.edge_points.remove(P)

		bins_image = self.helper.bins_image

		# Now remove the counts
		self.fg_sum -= self.helper.in_image[y, x]
		self.count -= 1
		bins_image[y, x] = -1

		if self.helper.back_image is not None:
			bs = self.helper.expmap_image[y, x] / self.helper.bg_expmap_image[y, x]
			bg = self.helper.back_image[y, x]

			self.bg_sum -= bg
			self.bg_sum_weight -= bg * bs
			self.expratio_sum_2 -= bs * bs

		if self.helper.noisemap_image is not None:
			self.noisemap_2_sum -= self.square(self.helper.noisemap_image[y, x])

		xw = self.helper.xw
		yw = self.helper.yw

		for n in range(self.bin_no_neigh):
			xp = x + self.bin_neigh_x[n]
			yp = y + self.bin_neigh_y[n]

			if 0 <= xp < xw and 0 <= yp < yw and bins_image[yp, xp] == self.bin_no:
				if (xp, yp) not in self.edge_points:
					self.edge_points.append((xp, yp))

	def paint_bins_image(self):
		"""Paint the bin number onto the bins image."""
		bins_image = self.helper.bins_image
		for (x, y) in self.all_points:
			bins_image[y, x] = self.bin_no

	def add_next_pixel(self):
		"""Add the next best pixel to the bin."""
		xw = self.helper.xw
		yw = self.helper.yw
		mask_image = self.helper.mask_image
		bins_image = self.helper.bins_image
		smoothed_image = self.helper.smoothed_image
		constrain_fill = self.helper.constrain_fill

		delta = 1e99
		bestx = -1
		besty = -1

		# List to hold indices of edge points to remove
		points_to_remove = []

		# Iterate over edge points
		for idx, (x, y) in enumerate(self.edge_points):
			is_edge = False

			for n in range(self.bin_no_neigh):
				xp = x + self.bin_neigh_x[n]
				yp = y + self.bin_neigh_y[n]

				if 0 <= xp < xw and 0 <= yp < yw:
					bin = bins_image[yp, xp]
					if bin != self.bin_no:
						is_edge = True

					if bin < 0 and mask_image[yp, xp] == 1:
						if (not constrain_fill) or self.check_constraint(xp, yp):
							newdelta = abs(smoothed_image[yp, xp] - self._aimval)
							if newdelta < delta:
								delta = newdelta
								bestx = xp
								besty = yp

			if not is_edge:
				points_to_remove.append(idx)

		# Remove non-edge points after iteration
		for idx in reversed(points_to_remove):
			del self.edge_points[idx]

		if bestx == -1:
			return False

		self.add_point(bestx, besty)
		return True
		
class Binner:
	"""Class responsible for performing the binning process."""
	def __init__(self, in_image, smoothed_image, threshold):
		self.xw = in_image.shape[1]
		self.yw = in_image.shape[0]
		self.bins_image = np.full((self.yw, self.xw), -1, dtype=int)
		self.binned_image = np.zeros((self.yw, self.xw))
		self.sn_image = np.zeros((self.yw, self.xw))
		self.bin_helper = BinHelper(in_image, smoothed_image, self.bins_image, threshold)
		self.bin_counter = 0
		self.bins = []
		self.sorted_pixels = []
		self.sorted_pix_posn = 0

	def set_back_image(self, back_image, expmap_image, bg_expmap_image):
		"""Set background images."""
		self.bin_helper.set_back(back_image, expmap_image, bg_expmap_image)

	def set_noisemap_image(self, noisemap_image):
		"""Set noise map image."""
		self.bin_helper.set_noisemap(noisemap_image)

	def set_mask_image(self, mask_image):
		"""Set mask image."""
		self.bin_helper.set_mask(mask_image)

	def set_constrain_fill(self, constrain_val):
		"""Set constraint fill value."""
		self.bin_helper.set_constrain_fill(constrain_val)

	def set_scrub_large_bins(self, fraction):
		"""Set scrub large bins fraction."""
		self.bin_helper.set_scrub_large_bins(fraction)

	def do_binning(self, bin_down):
		"""Perform the binning process."""
		# sort pixels into flux order to find starting pixels
		self.sort_pixels(bin_down)

		in_image = self.bin_helper.in_image
		in_back = self.bin_helper.back_image

		# safety check for dimensions of images
		assert self.bins_image.shape == (self.yw, self.xw)
		if in_back is not None:
			assert in_back.shape == (self.yw, self.xw)
		assert in_image.shape == (self.yw, self.xw)
		assert self.sn_image.shape == (self.yw, self.xw)
		assert self.binned_image.shape == (self.yw, self.xw)

		print("Starting binning")

		pix_counter = 0  # how many pixels processed
		no_unmasked = self.no_unmasked_pixels()

		# get next pixel
		nextpoint = self.find_next_pixel()
		assert nextpoint[0] >= 0 and nextpoint[1] >= 0

		# repeat binning, adding centroids and weights of bins
		while nextpoint[0] >= 0 and nextpoint[1] >= 0:
			# progress counter
			counter = self.bin_helper.bin_counter
			if counter % 10 == 0 and counter > 0:
				print("{:5d} ".format(counter), end='')
				sys.stdout.flush()
				if counter % 100 == 0:
					print(" [{:.1f}%]".format(pix_counter * 100. / no_unmasked))

			# make the new bin and do the binning
			newbin = Bin(self.bin_helper)
			newbin.do_binning(nextpoint[0], nextpoint[1])
			self.bins.append(newbin)

			# keep track of all the pixels binned
			pix_counter += newbin.count

			# find the next pixel
			nextpoint = self.find_next_pixel()

		self.bin_counter = self.bin_helper.bin_counter

		print(" [100.0%]")
		print(" Done binning ({} bins)".format(self.bin_counter))

	def sort_pixels(self, bin_down):
		if bin_down:
			print("Sorting pixels, binning from top...")
		else:
			print("Sorting pixels, binning from bottom...")
		sys.stdout.flush()

		in_mask = self.bin_helper.mask_image
		self.sorted_pixels = []

		for y in range(self.yw):
			for x in range(self.xw):
				if in_mask[y, x] >= 1:
					self.sorted_pixels.append((x, y))

		smoothed_image = self.bin_helper.smoothed_image

		# Define a key function for sorting
		def sort_key(p):
			x, y = p
			return smoothed_image[y, x]

		# Now sort the pixels
		self.sorted_pixels.sort(key=sort_key, reverse=bin_down)

		# iterator position
		self.sorted_pix_posn = 0

		print(" Done.")

	def find_next_pixel(self):
		"""Find the next unbinned pixel."""
		in_bins = self.bin_helper.bins_image

		while self.sorted_pix_posn < len(self.sorted_pixels):
			x, y = self.sorted_pixels[self.sorted_pix_posn]

			if in_bins[y, x] < 0:
				return (x, y)

			self.sorted_pix_posn += 1

		return (-1, -1)

	def no_unmasked_pixels(self):
		"""Count the number of unmasked pixels."""
		in_mask = self.bin_helper.mask_image
		no_unmasked = np.sum(in_mask >= 1)
		return no_unmasked

	def get_output_image(self):
		"""Get the output binned image."""
		return self.binned_image

	def get_binmap_image(self):
		"""Get the bin map image."""
		return self.bins_image

	def get_sn_image(self):
		"""Get the signal-to-noise image."""
		return self.sn_image

	def do_scrub(self):
		"""Perform the scrubbing process after binning."""
		scrubber = Scrubber(self.bin_helper, self.bins)
		scrubber.scrub()

		if self.bin_helper.scrub_large_bins > 0.0:
			scrubber.scrub_large_bins(self.bin_helper.scrub_large_bins)

		scrubber.renumber()

	def calc_outputs(self, output_dir='.'):
		"""Calculate the output images."""
		no_bins = len(self.bins)
		signal = [0.0] * no_bins
		noise_2 = [0.0] * no_bins
		pixcounts = [0] * no_bins
		sn = [0.0] * no_bins

		min_sn = float('inf')
		max_sn = float('-inf')
		min_signal = float('inf')
		max_signal = float('-inf')

		# Iterate over bins & collect info
		for i in range(no_bins):
			b = self.bins[i]
			bin_no = b.bin_no
			if bin_no < 0:
				continue

			assert bin_no < no_bins

			signal[bin_no] = b.signal()
			max_signal = max(signal[bin_no], max_signal)
			min_signal = min(signal[bin_no], min_signal)

			noise_2[bin_no] = b.noise_2()
			pixcounts[bin_no] = b.count

			sn[bin_no] = math.sqrt(b.sn_2())

			if not math.isfinite(sn[bin_no]) or sn[bin_no] < 0:
				print("WARNING: Invalid value in signal-to-noise. "
					  "This can be caused by a negative input image.")

			max_sn = max(sn[bin_no], max_sn)
			min_sn = min(sn[bin_no], min_sn)

		# Now make output images
		self.sn_image.fill(-1)
		self.binned_image.fill(-1)
		bins_image = self.bins_image

		for y in range(self.yw):
			for x in range(self.xw):
				bin_num = bins_image[y, x]
				if bin_num >= 0:
					self.sn_image[y, x] = sn[bin_num]
					self.binned_image[y, x] = signal[bin_num] / pixcounts[bin_num]

		# build histogram of signal to noises
		no_hbins = 30
		delta_sn = (max_sn - min_sn + 0.0001) / no_hbins
		delta_signal = (max_signal - min_signal + 0.0001) / no_hbins
		histo_sn = [0] * no_hbins
		histo_signal = [0] * no_hbins

		for bin_num in range(no_bins):
			if self.bins[bin_num].bin_no < 0:
				continue

			index_sn = int((sn[bin_num] - min_sn) / delta_sn)
			index_signal = int((signal[bin_num] - min_signal) / delta_signal)

			if index_sn >= no_hbins:
				index_sn = no_hbins - 1
			if index_signal >= no_hbins:
				index_signal = no_hbins - 1

			assert index_sn < no_hbins and index_signal < no_hbins

			histo_sn[index_sn] += 1
			histo_signal[index_signal] += 1

		sn_qdp_path = os.path.join(output_dir, "bin_sn_stats.qdp")
		signal_qdp_path = os.path.join(output_dir, "bin_signal_stats.qdp")

		# output histogram data in file
		with open(sn_qdp_path, "w") as stream_sn, \
			 open(signal_qdp_path, "w") as stream_signal:

			stream_sn.write(
				"label x Signal:Noise\n"
				"label y Number\n"
				"line step\n"
			)
			stream_signal.write(
				"label x Counts\n"
				"label y Number\n"
				"line step\n"
			)

			# write out histograms
			for h in range(no_hbins):
				stream_sn.write("{:.6f}\t{}\n".format(min_sn + (h + 0.5) * delta_sn, histo_sn[h]))
				stream_signal.write("{:.6f}\t{}\n".format(min_signal + (h + 0.5) * delta_signal, histo_signal[h]))
				
class Scrubber:
	"""Class responsible for scrubbing the bins after binning."""
	def __init__(self, helper, bins):
		self._helper = helper
		self.bins = bins
		self.no_bins = len(self.bins)
		self.scrub_sn_2 = self.square(helper.threshold)
		self.cannot_dissolve = [False] * self.no_bins
		self.xw = helper.xw
		self.yw = helper.yw
		self.bin_no_neigh = self._helper.bin_no_neigh
		self.bin_neigh_x = self._helper.bin_neigh_x
		self.bin_neigh_y = self._helper.bin_neigh_y


	@staticmethod
	def square(d):
		return d * d

	def find_best_neighbour(self, thebin, allow_unconstrained):
		"""Find the best neighboring bin to dissolve into."""
		smoothed_image = self._helper.smoothed_image
		bins_image = self._helper.bins_image
		binno = thebin.bin_no

		# bestdelta = 1e99
		bestdelta = float('inf')
		bestx = -1
		besty = -1
		bestbin = -1

		edgepoints = thebin.get_edge_points()
		pt = 0

		while pt < len(edgepoints):
			x, y = edgepoints[pt]
			v = smoothed_image[y, x]

			# loop over neighbours of edge point
			anyneighbours = False
			for n in range(self.bin_no_neigh):
				xp = x + self.bin_neigh_x[n]
				yp = y + self.bin_neigh_y[n]

				# select pixels in neighbouring bins
				if 0 <= xp < self.xw and 0 <= yp < self.yw:
					# if the neighbour is a real and different bin
					nbin = bins_image[yp, xp]
					if nbin != -1 and nbin != binno:
						anyneighbours = True

						# skip neighbours with too long a constraint if required
						if self._helper.constrain_fill and not allow_unconstrained and not self.bins[nbin].check_constraint(xp, yp):
							continue

						delta = abs(v - smoothed_image[yp, xp])
						if delta < bestdelta:
							bestdelta = delta
							bestx = x
							besty = y
							bestbin = nbin

			# remove edge pixels without any neighbours
			if not anyneighbours:
				edgepoints.pop(pt)
			else:
				pt += 1

		return bestx, besty, bestbin

	def dissolve_bin(self, thebin):
		"""Dissolve a bin into its neighbors."""
		# loop until no pixels remaining
		while thebin.count != 0:
			bestx, besty, bestbin = self.find_best_neighbour(thebin, allow_unconstrained=False)

			# if none, then ignore constraints
			if bestx == -1 and self._helper.constrain_fill:
				bestx, besty, bestbin = self.find_best_neighbour(thebin, allow_unconstrained=True)

			# stop dissolving bin if we have no neighbours for our remaining pixels
			if bestbin == -1:
				binno = thebin.bin_no
				print(f"WARNING: Could not dissolve bin {binno} into surroundings")
				self.cannot_dissolve[binno] = True
				return

			# Ensure we do not create excessively large bins
			# if self.bins[bestbin].count() > 0.1 * (self.xw * self.yw):  # Example: set max bin size to 10%
			# 	print(f"WARNING: Bin {bestbin} is too large to merge into")
			# 	continue
		
			# reassign pixel
			thebin.remove_point(bestx, besty)
			self.bins[bestbin].add_point(bestx, besty)

	def scrub(self):
		"""Perform the scrubbing process."""
		print("Starting scrubbing...")

		# put bins into a pointer array so we can discard them quickly when we don't need to consider them
		bin_ptrs = [bin for bin in self.bins if bin.sn_2() < self.scrub_sn_2]

		# we keep looping until the lowest S/N bin is removed
		while True:
			lowest_SN_2 = 1e99
			lowest_bin = None

			i = 0
			while i < len(bin_ptrs):
				SN_2 = bin_ptrs[i].sn_2()
				# if this bin has a larger S/N than threshold, remove it
				if SN_2 >= self.scrub_sn_2:
					bin_ptrs.pop(i)
				else:
					# if this is lower than before, store it
					if SN_2 < lowest_SN_2:
						lowest_SN_2 = SN_2
						lowest_bin = bin_ptrs[i]
					i += 1

			# exit if no more bins remaining
			if lowest_bin is None or lowest_SN_2 >= self.scrub_sn_2:
				break

			# get rid of that bin (if it cannot be dissolved, it doesn't matter)
			self.dissolve_bin(lowest_bin)
			bin_ptrs.remove(lowest_bin)

			# show progress to user
			if len(bin_ptrs) % 10 == 0:
				print(f"{len(bin_ptrs):5d} ", end='')
				if len(bin_ptrs) % 100 == 0:
					print()

		print(" Done.")

	def scrub_large_bins(self, fraction=0.05): # =0.05
		"""Scrub bins that are too large."""
		print(f"Scrubbing bins with fraction of area > {fraction}...")

		# get total number of pixels in bins
		totct = sum(bin.count for bin in self.bins)

		# now get rid of large bins
		for bin in self.bins:
			thisfrac = bin.count / totct
			if thisfrac >= fraction:
				print(f" Scrubbing bin {bin.bin_no}")
				bin.drop_bin()

	def renumber(self):
		"""Renumber bins after scrubbing."""
		print("\nStarting renumbering...")

		# split bins into those with counts and those without
		self.bins = [bin for bin in self.bins if bin.count > 0]

		# now clear bin image, and repaint everything (doing renumber)
		self._helper.bins_image.fill(-1)

		number = 0
		for bin in self.bins:
			bin.set_bin_no(number)
			bin.paint_bins_image()
			number += 1

		print(f"{number} bins when finished\n Done.")

