try:
    import os
    import numpy as np
    from scipy.ndimage import convolve
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS
    from astropy.nddata import Cutout2D
    from astropy import units as u
    from astropy.stats import sigma_clipped_stats
    from photutils.segmentation import SegmentationImage
    from photutils.segmentation import make_2dgaussian_kernel, detect_sources, deblend_sources
    from photutils.background import Background2D, SExtractorBackground

    # Set environment variables
    os.environ["OMP_NUM_THREADS"] = "1"
    
except ImportError as e:
    print(f"Error: {e}") 

def fill_masked_pixels(image):
    """
    Fills NaN or zero-valued pixels in an image using background noise sampled from the data.
    
    Parameters:
    - image: 2D numpy array (grayscale), the masked image with NaN or 0 values.

    Returns:
    - filled_image: 2D numpy array with the masked regions replaced by interpolated noise.
    """
    filled_image = image.copy()

    # Identify masked pixels (NaN or zero)
    mask = np.isnan(image) | (image == 0)

    # Check if there are any masked pixels
    if not np.any(mask):
        return image  # No missing pixels, return original image

    # Extract background pixel values (valid data)
    background_values = image[~mask]

    if len(background_values) == 0:
        raise ValueError("No valid background pixels found in the image!")

    # Estimate background noise characteristics
    mean_bg = np.mean(background_values)
    std_bg = np.std(background_values)

    # Generate noise sampled from the image background
    noise = np.random.normal(mean_bg, std_bg, size=image.shape)

    # Fill missing pixels with local noise
    filled_image[mask] = noise[mask]

    return filled_image

def cutout_centered_on_local_peak_with_tolerance(
    image_data_filt0, wts_data_filt0, image_data_filt1, wts_data_filt1, position, size, tolerance_arcsec):
    """
    Create a cutout image around a specified RA/Dec, find the peak flux within a tolerance 
    angular size, recenter a cutout around the peak, and optionally calculate RMS noise.

    Parameters:
    - image_data_filt0, wts_data_filt0: 2D arrays for the first dataset (flux and weights).
    - image_data_filt1, wts_data_filt1: 2D arrays for the second dataset (flux and weights).
    - position (tuple): RA and Dec of the centre to look around, in degrees.
    - size (int or tuple): Cutout size, in pixels (ny, nx).
    - tolerance_arcsec (float): Angular size tolerance (in arcseconds) to find the peak around the centre.

    Returns:
    - peak_cutout_filt0: Cutout of image_data_filt0 centred on the peak flux.
    - peak_cutout_wts0: Cutout of wts_data_filt0 centred on the peak flux.
    - peak_cutout_filt1: Cutout of image_data_filt1 centred on the peak flux.
    - peak_cutout_wts1: Cutout of wts_data_filt1 centred on the peak flux.
    """
    # Step 1: Convert RA/Dec to pixel coordinates for the initial centre
    coord = SkyCoord(ra=position[0] * u.deg, dec=position[1] * u.deg, frame='icrs')
    wcs = WCS(image_data_filt0[0])
    initial_center_pix = wcs.world_to_pixel(coord)

    # Step 2: Create an initial cutout around the specified RA/Dec position
    initial_cutout = Cutout2D(image_data_filt0[0].data, initial_center_pix, size, wcs=wcs)

    # Step 3: Convert the tolerance from arcseconds to pixels
    pixel_scale = np.abs(wcs.pixel_scale_matrix[1, 1]) * 3600  # arcseconds per pixel
    tolerance_pixels = tolerance_arcsec / pixel_scale

    # Step 4: Create a circular mask within the tolerance radius
    ny, nx = initial_cutout.data.shape
    y, x = np.ogrid[:ny, :nx]
    center_y, center_x = ny // 2, nx // 2
    distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    mask = distance_from_center <= tolerance_pixels

    # Step 5: Find the peak flux within the masked region
    masked_data = np.where(mask, initial_cutout.data, -np.inf)
    peak_position_cutout = np.unravel_index(np.argmax(masked_data), masked_data.shape)

    # Debug: Verify peak position in the cutout
    print(f"Peak position in cutout: {peak_position_cutout}")

    # Calculate global pixel coordinates of the peak position
    peak_position_global = (
        initial_cutout.origin_original[0] + peak_position_cutout[1],
        initial_cutout.origin_original[1] + peak_position_cutout[0],
    )

    # Step 6: Create new cutouts centred on the peak position
    peak_cutout_filt0 = Cutout2D(image_data_filt0[0].data, peak_position_global, size, wcs=wcs)
    peak_cutout_wts0 = Cutout2D(wts_data_filt0[0].data, peak_position_global, size, wcs=wcs)
    peak_cutout_filt1 = Cutout2D(image_data_filt1[0].data, peak_position_global, size, wcs=wcs)
    peak_cutout_wts1 = Cutout2D(wts_data_filt1[0].data, peak_position_global, size, wcs=wcs)

    return peak_cutout_filt0, peak_cutout_wts0, peak_cutout_filt1, peak_cutout_wts1

def get_cutout_data_array_v2(sample_data, filters, size, use_control=False):
    """
    Generate cutouts centered on the local peak within a specified tolerance around RA/Dec positions.
    
    Parameters:
    - sample_data (structured array or DataFrame): Input data with RA/Dec values.
    - use_control (bool): If True, use column names 'RA' and 'dec'; otherwise, use 'ra' and 'dec'.
    
    Returns:
    - peak_cutout_filt0_arr (list): Cutouts of filtered image data (filter 0).
    - peak_cutout_wts0_arr (list): Cutouts of weights data (filter 0).
    - peak_cutout_filt1_arr (list): Cutouts of filtered image data (filter 1).
    - peak_cutout_wts1_arr (list): Cutouts of weights data (filter 1).
    - err_ind (numpy array): Indices of failed cutouts.
    """
    # Initialise result lists
    peak_cutout_filt0_arr, peak_cutout_wts0_arr = [], []
    peak_cutout_filt1_arr, peak_cutout_wts1_arr = [], []
    err_ind = []

    # Column selection based on `use_control`
    ra_col = 'ra' if use_control else 'ra'
    dec_col = 'dec'

    # Iterate through each entry in sample_data
    for i in range(len(sample_data)):
        try:
            # Extract RA/Dec for the current index
            position = [sample_data[ra_col][i], sample_data[dec_col][i]]
            f0, fw0, f1, fw1 = filters
            # Call the cutout function with tolerance
            peak_cutout_filt0, peak_cutout_wts0, peak_cutout_filt1, peak_cutout_wts1 = \
                cutout_centered_on_local_peak_with_tolerance(
                    image_data_filt0=f0,
                    wts_data_filt0=fw0,
                    image_data_filt1=f1,
                    wts_data_filt1=fw1,
                    position=position,
                    size=size * u.arcsecond,
                    tolerance_arcsec=0.1
                )
            
            # Append the resulting cutouts
            peak_cutout_filt0_arr.append(peak_cutout_filt0)
            peak_cutout_wts0_arr.append(peak_cutout_wts0)
            peak_cutout_filt1_arr.append(peak_cutout_filt1)
            peak_cutout_wts1_arr.append(peak_cutout_wts1)
        
        except Exception as e:
            # Print specific error details and log the index
            print(f"Error at index {i}: {e}")
            err_ind.append(i)
    
    # Convert error indices to numpy array for consistency
    err_ind = np.array(err_ind)
    
    return peak_cutout_filt0_arr, peak_cutout_wts0_arr, peak_cutout_filt1_arr, peak_cutout_wts1_arr, err_ind


def get_wcs_pixel_scale(wcs):
    """Returns the pixel scale in arcseconds per pixel."""
    return np.abs(wcs.pixel_scale_matrix[1, 1]) * 3600

def create_cutout(image_data, wcs, position, size):
    """Creates a cutout from an image centered on a given position."""
    return Cutout2D(image_data, position, size, wcs=wcs)

def find_peak_position(cutout):
    """Finds the peak flux position in the cutout and returns its global coordinates."""
    peak_local = np.unravel_index(np.argmax(cutout.data), cutout.data.shape)
    return (cutout.origin_original[0] + peak_local[1], cutout.origin_original[1] + peak_local[0])

def cutout_centered_on_local_peak(
    image_data_filt0, wts_data_filt0, image_data_filt1, wts_data_filt1,
    position, size, tolerance_arcsec=None):
    """
    Creates cutouts centered on the peak flux near the specified RA/Dec position.
    If tolerance_arcsec is provided, it finds the peak flux within that region.
    """
    coord = SkyCoord(ra=position[0] * u.deg, dec=position[1] * u.deg, frame='icrs')
    wcs = WCS(image_data_filt0[0])
    initial_center_pix = wcs.world_to_pixel(coord)

    # Create an initial cutout
    initial_cutout = create_cutout(image_data_filt0[0].data, wcs, initial_center_pix, size)
    
    if tolerance_arcsec:
        pixel_scale = get_wcs_pixel_scale(wcs)
        tolerance_pixels = tolerance_arcsec / pixel_scale
        ny, nx = initial_cutout.data.shape
        y, x = np.ogrid[:ny, :nx]
        mask = np.sqrt((x - nx // 2) ** 2 + (y - ny // 2) ** 2) <= tolerance_pixels
        masked_data = np.where(mask, initial_cutout.data, -np.inf)
    else:
        masked_data = initial_cutout.data
    
    peak_position = find_peak_position(initial_cutout)
    
    return (
        create_cutout(image_data_filt0[0].data, wcs, peak_position, size),
        create_cutout(wts_data_filt0[0].data, wcs, peak_position, size),
        create_cutout(image_data_filt1[0].data, wcs, peak_position, size),
        create_cutout(wts_data_filt1[0].data, wcs, peak_position, size)
    )

def cutout_with_background_rms(
    image_data_filt0, wts_data_filt0, image_data_filt1, wts_data_filt1,
    position, size, calculate_rms=False, annulus_distance_arcsec=40, annulus_width_arcsec=20):
    """
    Creates cutouts centered on the peak flux and optionally calculates background RMS noise.
    """
    cutouts = cutout_centered_on_local_peak(image_data_filt0, wts_data_filt0, image_data_filt1, wts_data_filt1, position, size)
    
    if not calculate_rms:
        return cutouts
    
    wcs = WCS(image_data_filt0[0])
    pixel_scale = get_wcs_pixel_scale(wcs)
    inner_radius, outer_radius = annulus_distance_arcsec / pixel_scale, (annulus_distance_arcsec + annulus_width_arcsec) / pixel_scale
    
    ny, nx = cutouts[0].data.shape
    y, x = np.ogrid[:ny, :nx]
    mask = (inner_radius <= np.sqrt((x - nx // 2) ** 2 + (y - ny // 2) ** 2)) & \
           (np.sqrt((x - nx // 2) ** 2 + (y - ny // 2) ** 2) <= outer_radius)
    
    background_rms = np.std(cutouts[0].data[mask])
    return cutouts + (background_rms,)

def fill_masked_with_noise(image, mask):
    """Fills masked regions of an image with Gaussian noise matching unmasked regions."""
    if image.shape != mask.shape:
        raise ValueError("Image and mask must have the same shape.")
    
    noise_level = np.std(image[~mask])
    return np.where(mask, np.random.normal(0, noise_level, image.shape), image)

def apply_closest_mask(data, segment: SegmentationImage):
    """Applies the closest mask to the centre and merges all other masks."""
    centre_y, centre_x = np.array(data.shape) // 2
    segment_masks = [(segment.data == label) for label in range(1, segment.nlabels + 1)]
    
    if segment.nlabels == 1:
        return np.where(segment_masks[0], data, np.nan), segment_masks[0], np.zeros_like(data, dtype=bool)
    
    distances = [np.min(np.sqrt((np.where(mask)[1] - centre_x) ** 2 + (np.where(mask)[0] - centre_y) ** 2)) for mask in segment_masks]
    closest_index = np.argmin(distances)
    closest_mask = segment_masks[closest_index]
    merged_other_masks = np.logical_or.reduce([mask for i, mask in enumerate(segment_masks) if i != closest_index])
    
    return np.where(closest_mask, data, np.nan), closest_mask, merged_other_masks

def get_cutout_data_array(sample_data, filters, use_control=False):
    """
    Process a list of RA/Dec positions and generate cutouts for each position.
    
    Parameters:
    - sample_data (structured array or DataFrame): Input data with RA/Dec values.
    - use_control (bool): If True, use column names 'RA' and 'dec'; otherwise, use 'ra' and 'dec'.
    
    Returns:
    - peak_cutout_filt0_arr (list): Cutouts of filtered image data (filter 0).
    - peak_cutout_wts0_arr (list): Cutouts of weights data (filter 0).
    - peak_cutout_filt1_arr (list): Cutouts of filtered image data (filter 1).
    - peak_cutout_wts1_arr (list): Cutouts of weights data (filter 1).
    - err_ind (numpy array): Indices of failed cutouts.
    """
    # Initialize result arrays
    peak_cutout_filt0_arr, peak_cutout_wts0_arr = [], []
    peak_cutout_filt1_arr, peak_cutout_wts1_arr = [], []
    err_ind = []

    # Select the column names for RA and Dec based on `use_control`
    ra_col = 'RA' if use_control else 'ra'
    dec_col = 'dec'
    f0, fw0, f1, fw1 = filters
    for i in range(len(sample_data)):
        try:
            # Extract RA/Dec for the current index
            position = [sample_data[ra_col][i], sample_data[dec_col][i]]
            
            # Generate cutouts
            peak_cutout_filt0, peak_cutout_wts0, peak_cutout_filt1, peak_cutout_wts1 = \
                cutout_centered_on_local_peak_with_tolerance(
                    image_data_filt0=f0,
                    wts_data_filt0=fw0,
                    image_data_filt1=f1,
                    wts_data_filt1=fw1,
                    position=position,
                    size=5 * u.arcsecond
                )
            
            # Append results
            peak_cutout_filt0_arr.append(peak_cutout_filt0)
            peak_cutout_wts0_arr.append(peak_cutout_wts0)
            peak_cutout_filt1_arr.append(peak_cutout_filt1)
            peak_cutout_wts1_arr.append(peak_cutout_wts1)
        
        except Exception as e:
            print(f"Error at index {i}: {e}")
            err_ind.append(i)
    
    # Convert error indices to numpy array
    err_ind = np.array(err_ind)
    
    return peak_cutout_filt0_arr, peak_cutout_wts0_arr, peak_cutout_filt1_arr, peak_cutout_wts1_arr, err_ind

def get_cleaned_data(arr, arr_ind):
    arr_dummy = []
    for i in range(len(arr_ind)):
        arr_dummy.append(arr[arr_ind[i]])
    
    return np.array(arr_dummy)

def get_cutout_data_array_v2(sample_data, filters, size, use_control=False):
    """
    Generate cutouts centered on the local peak within a specified tolerance around RA/Dec positions.
    
    Parameters:
    - sample_data (structured array or DataFrame): Input data with RA/Dec values.
    - use_control (bool): If True, use column names 'RA' and 'dec'; otherwise, use 'ra' and 'dec'.
    
    Returns:
    - peak_cutout_filt0_arr (list): Cutouts of filtered image data (filter 0).
    - peak_cutout_wts0_arr (list): Cutouts of weights data (filter 0).
    - peak_cutout_filt1_arr (list): Cutouts of filtered image data (filter 1).
    - peak_cutout_wts1_arr (list): Cutouts of weights data (filter 1).
    - err_ind (numpy array): Indices of failed cutouts.
    """
    # Initialise result lists
    peak_cutout_filt0_arr, peak_cutout_wts0_arr = [], []
    peak_cutout_filt1_arr, peak_cutout_wts1_arr = [], []
    err_ind = []

    # Column selection based on `use_control`
    ra_col = 'ra' if use_control else 'ra'
    dec_col = 'dec'
    f0, fw0, f1, fw1 = filters
    # Iterate through each entry in sample_data
    for i in range(len(sample_data)):
        try:
            # Extract RA/Dec for the current index
            position = [sample_data[ra_col][i], sample_data[dec_col][i]]
            
            # Call the cutout function with tolerance
            peak_cutout_filt0, peak_cutout_wts0, peak_cutout_filt1, peak_cutout_wts1 = \
                cutout_centered_on_local_peak_with_tolerance(
                    image_data_filt0=f0,
                    wts_data_filt0=fw0,
                    image_data_filt1=f1,
                    wts_data_filt1=fw1,
                    position=position,
                    size=size * u.arcsecond,
                    tolerance_arcsec=0.1
                )
            
            # Append the resulting cutouts
            peak_cutout_filt0_arr.append(peak_cutout_filt0)
            peak_cutout_wts0_arr.append(peak_cutout_wts0)
            peak_cutout_filt1_arr.append(peak_cutout_filt1)
            peak_cutout_wts1_arr.append(peak_cutout_wts1)
        
        except Exception as e:
            # Print specific error details and log the index
            print(f"Error at index {i}: {e}")
            err_ind.append(i)
    
    # Convert error indices to numpy array for consistency
    err_ind = np.array(err_ind)
    
    return peak_cutout_filt0_arr, peak_cutout_wts0_arr, peak_cutout_filt1_arr, peak_cutout_wts1_arr, err_ind

def get_segments_background(peak_cutout_data_arr, box_size_, filt_size, npixels, nlevels, contrast):  
    bkg_estimator = SExtractorBackground()
    segment_map_arr = []
    data_sub_arr = []
    convolved_data_arr = []
    err_ind = []
    for i in range(len(peak_cutout_data_arr)):
        try:
            bkg = Background2D(peak_cutout_data_arr[i], box_size=(box_size_, box_size_),\
                               filter_size=(filt_size, filt_size),
                               bkg_estimator=bkg_estimator)
            data_sub_arr.append(peak_cutout_data_arr[i] - bkg.background)  # subtract the background
            
            kernel = make_2dgaussian_kernel(2.5, size=3)  # FWHM = 3.0
            convolved_data_arr.append(convolve(data_sub_arr[i], kernel))
            mean, median, std = sigma_clipped_stats(data_sub_arr[i], sigma=3.0) 
    
            segment_map_arr.append(detect_sources(convolved_data_arr[i], threshold=std, npixels=10))
        except Exception as e:
            err_ind.append(i)
            print(f"Error in segment creation at index {i}: {e}")
    
    segm_deblend_arr = []
    for i in range(len(peak_cutout_data_arr)):
        try:
            segm_deblend_arr.append(deblend_sources(convolved_data_arr[i], segment_map_arr[i],
                                                    npixels=npixels, nlevels=nlevels, contrast=contrast,
                                                    progress_bar=False))
        except Exception as e:
            err_ind.append(i)
            print(f"Error in deblending at index {i}: {e}")
            
    background_filled_arr = []
    for i in range(len(peak_cutout_data_arr)):
        try:
            background_filled_arr.append(fill_masked_with_noise(image=data_sub_arr[i],
                                                                 mask=~segm_deblend_arr[i].data_ma.mask))
        except Exception as e:
            err_ind.append(i)
            print(f"Error in background filling at index {i}: {e}")
    
    masked_image_arr, closest_mask_arr, merged_other_masks_arr = [], [], []
    for i in range(len(peak_cutout_data_arr)):
        try:
            masked_image, closest_mask, merged_other_masks = apply_closest_mask(data=data_sub_arr[i],
                                                                                segment=segm_deblend_arr[i])
            masked_image_arr.append(masked_image)
            closest_mask_arr.append(closest_mask)
            merged_other_masks_arr.append(merged_other_masks)
        except Exception as e:
            err_ind.append(i)
            print(f"Error in masking at index {i}: {e}")
            
    background_filled_arr2 = []
    for i in range(len(peak_cutout_data_arr)):
        try:
            background_filled_arr2.append(fill_masked_with_noise(image=data_sub_arr[i],
                                                                  mask=merged_other_masks_arr[i]))
        except Exception as e:
            err_ind.append(i)
            print(f"Error in second background filling at index {i}: {e}")
            
    background_filled_arr3 = []
    for i in range(len(peak_cutout_data_arr)):
        try:
            background_filled_arr3.append(np.ma.masked_array(background_filled_arr[i], 
                                                              mask=closest_mask_arr[i], fill_value=np.nan))
        except Exception as e:
            err_ind.append(i)
            print(f"Error in third background filling at index {i}: {e}")
    err_ind = np.unique(np.array(err_ind))
    return segm_deblend_arr, segment_map_arr, data_sub_arr, convolved_data_arr, closest_mask_arr, background_filled_arr2, background_filled_arr3, err_ind

def process_array(arr_list, bad_ind, target_shape):
    """
    Processes an array list by setting `bad_ind` to NaN and then restoring its data.

    Parameters:
    - arr_list: List of arrays to process.
    - bad_ind: Index of the bad data to process.
    - target_shape: Target shape for the NaN-filled array.

    Returns:
    - Updated list of arrays.
    """
    # Extract data
    arr_data = [arr.data for arr in arr_list]

    # Backup and replace with NaNs
    temp_data = np.copy(arr_data[bad_ind])
    arr_data[bad_ind] = np.full(target_shape, np.nan)

    # Restore the original data into the NaN-filled array
    arr_data[bad_ind][:temp_data.shape[0], :temp_data.shape[1]] = temp_data

    return arr_data

def mask_highz_galaxies(data, segment, wcs, galaxy_ra, galaxy_dec, galaxy_redshifts, z_lim=3.08):
    """
    Masks galaxies with redshift > z_lim based on their RA/Dec coordinates in the WCS image.

    Parameters:
        data (numpy.ndarray): The 2D image array.
        segment (photutils.segmentation.SegmentationImage): The segmentation object with labeled masks.
        wcs (astropy.wcs.WCS): The WCS object corresponding to the data and segment images.
        galaxy_ra (array-like): RA values of the galaxies in degrees.
        galaxy_dec (array-like): Dec values of the galaxies in degrees.
        galaxy_redshifts (array-like): Redshift values of the galaxies.
        z_lim (float, optional): The redshift threshold to mask galaxies. Default is 3.08.

    Returns:
        masked_image (numpy.ndarray): The image with all high-redshift galaxies masked.
        combined_highz_mask (numpy.ndarray): A combined mask of all high-redshift galaxies.
        other_segments_mask (numpy.ndarray): A combined mask of all segments not included in the high-z mask.
    """
    # Convert galaxy RA/Dec to pixel coordinates using WCS
    galaxies_coords = SkyCoord(ra=galaxy_ra, dec=galaxy_dec, unit="deg")
    galaxy_x, galaxy_y = wcs.world_to_pixel(galaxies_coords)

    # Identify galaxies with redshift > z_lim
    if len(galaxy_redshifts) > 1:
        high_z_indices = np.where(galaxy_redshifts > z_lim)[0]
        high_z_x, high_z_y = galaxy_x[high_z_indices], galaxy_y[high_z_indices]
    else:
        high_z_x, high_z_y = galaxy_x, galaxy_y
    # Initialise a combined mask for high-z galaxies
    combined_highz_mask = np.zeros_like(segment.data, dtype=bool)

    # Loop through all high-redshift galaxies
    for x, y in zip(high_z_x, high_z_y):
        # Convert pixel coordinates to integer and ensure they are within image bounds
        x, y = int(round(x)), int(round(y))
        if 0 <= x < data.shape[1] and 0 <= y < data.shape[0]:
            # Find the segment label corresponding to this pixel
            label = segment.data[y, x]
            if label > 0:  # Exclude background (label 0)
                # Add the current segment to the combined high-z mask
                combined_highz_mask |= (segment.data == label)

    # Create the masked image by setting high-z galaxy regions to NaN
    masked_image = np.where(combined_highz_mask, np.nan, data)

    # Create a mask for all other segments not included in the high-z mask
    other_segments_mask = (segment.data > 0) & ~combined_highz_mask

    return masked_image, combined_highz_mask, other_segments_mask

def fill_masked_with_noise2(image, source_mask, background_mask):
    """
    Fills masked regions of an image with noise matching the unmasked regions.

    Parameters:
    - image (numpy.ndarray): The input image.
    - source_mask (numpy.ndarray): A boolean mask for source regions.
    - background_mask (numpy.ndarray): A boolean mask for background regions.

    Returns:
    - numpy.ndarray: The image with masked regions filled with noise.
    """
    if image.shape != source_mask.shape or image.shape != background_mask.shape:
        raise ValueError("Image and masks must have the same shape.")

    # Combine masks
    combined_mask = source_mask | background_mask
    
    # Create a masked array
    masked_image = np.ma.masked_array(image, mask=combined_mask)
    
    # Calculate noise level (standard deviation of unmasked pixels)
    noise_level = np.nanstd(masked_image)
    noise_loc = np.nanmean(masked_image)
    
    # Generate noise for masked pixels
    random_noise = np.random.normal(loc=noise_loc, scale=noise_level, size=image.shape)
    
    # Create a copy of the original image to modify
    noisy_image = image.copy()
    noisy_image[combined_mask] = random_noise[combined_mask]
    image[background_mask] = random_noise[background_mask]
    
    return noisy_image, image

def process_images_with_masks(data_arrays, source_masks, background_masks, good_indices, fill_masked_function, filt=True):
    """
    Applies a masking function to a list of data arrays, separating noise and sources.

    Parameters:
    - data_arrays: List of input data arrays to process.
    - source_masks: List of source masks corresponding to each array.
    - background_masks: List of background masks corresponding to each array.
    - good_indices: Indices of arrays to process.
    - fill_masked_function: Function for filling masked regions with noise and extracting sources.

    Returns:
    - img_only_noise_list: List of images with only noise.
    - img_with_source_list: List of images with sources included.
    """
    img_only_noise_list = []
    img_with_source_list = []

    for i in good_indices:
        data = data_arrays[i].data if filt else data_arrays[i]
        try:
            img_only_noise, img_with_source = fill_masked_function(
                image=data,
                source_mask=source_masks[i],
                background_mask=background_masks[i]
            )
            img_only_noise_list.append(img_only_noise)
            img_with_source_list.append(img_with_source)
        except Exception as e:
            print(f"Error processing index {i}: {e}")
            continue

    return img_only_noise_list, img_with_source_list