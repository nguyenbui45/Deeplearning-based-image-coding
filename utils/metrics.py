import torch
from torch import nn
import numpy as np
from PIL import Image
from scipy.ndimage import convolve
from PIL import Image
from scipy import signal

from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchvision import transforms
import math
from torch.distributions import Normal
from mean_average_precision import MetricBuilder


def _FSpecialGauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function."""
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()


def _SSIMForMultiScale(img1,
                       img2,
                       max_val=255,
                       filter_size=11,
                       filter_sigma=1.5,
                       k1=0.01,
                       k2=0.03):
    """Return the Structural Similarity Map between `img1` and `img2`.

  This function attempts to match the functionality of ssim_index_new.m by
  Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

  Arguments:
    img1: Numpy array holding the first RGB image batch.
    img2: Numpy array holding the second RGB image batch.
    max_val: the dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Size of blur kernel to use (will be reduced for small images).
    filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
      for small images).
    k1: Constant used to maintain stability in the SSIM calculation (0.01 in
      the original paper).
    k2: Constant used to maintain stability in the SSIM calculation (0.03 in
      the original paper).

  Returns:
    Pair containing the mean SSIM and contrast sensitivity between `img1` and
    `img2`.

  Raises:
    RuntimeError: If input images don't have the same shape or don't have four
      dimensions: [batch_size, height, width, depth].
  """
    if img1.shape != img2.shape:
        raise RuntimeError(
            'Input images must have the same shape (%s vs. %s).', img1.shape,
            img2.shape)
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d',
                           img1.ndim)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    _, height, width, _ = img1.shape

    # Filter size can't be larger than height or width of images.
    size = min(filter_size, height, width)

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:
        window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
        mu1 = signal.fftconvolve(img1, window, mode='valid')
        mu2 = signal.fftconvolve(img2, window, mode='valid')
        sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
        sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
        sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
    cs = np.mean(v1 / v2)
    return ssim, cs


def MultiScaleSSIM(img1,
                   img2,
                   max_val=255,
                   filter_size=11,
                   filter_sigma=1.5,
                   k1=0.01,
                   k2=0.03,
                   weights=None):
    """Return the MS-SSIM score between `img1` and `img2`.

  This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
  Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
  similarity for image quality assessment" (2003).
  Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf

  Author's MATLAB implementation:
  http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

  Arguments:
    img1: Numpy array holding the first RGB image batch.
    img2: Numpy array holding the second RGB image batch.
    max_val: the dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Size of blur kernel to use (will be reduced for small images).
    filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
      for small images).
    k1: Constant used to maintain stability in the SSIM calculation (0.01 in
      the original paper).
    k2: Constant used to maintain stability in the SSIM calculation (0.03 in
      the original paper).
    weights: List of weights for each level; if none, use five levels and the
      weights from the original paper.

  Returns:
    MS-SSIM score between `img1` and `img2`.

  Raises:
    RuntimeError: If input images don't have the same shape or don't have four
      dimensions: [batch_size, height, width, depth].
  """
    if img1.shape != img2.shape:
        raise RuntimeError(
            'Input images must have the same shape (%s vs. %s).', img1.shape,
            img2.shape)
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d',
                           img1.ndim)

    # Note: default weights don't sum to 1.0 but do match the paper / matlab code.
    weights = np.array(weights if weights else
                       [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size
    downsample_filter = np.ones((1, 2, 2, 1)) / 4.0
    im1, im2 = [x.astype(np.float64) for x in [img1, img2]]
    mssim = np.array([])
    mcs = np.array([])
    for _ in range(levels):
        ssim, cs = _SSIMForMultiScale(
            im1,
            im2,
            max_val=max_val,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1,
            k2=k2)
        mssim = np.append(mssim, ssim)
        mcs = np.append(mcs, cs)
        filtered = [
            convolve(im, downsample_filter, mode='reflect')
            for im in [im1, im2]
        ]
        im1, im2 = [x[:, ::2, ::2, :] for x in filtered]
    return (np.prod(mcs[0:levels - 1]**weights[0:levels - 1]) *
            (mssim[levels - 1]**weights[levels - 1]))


def msssim(original, compared):
    if isinstance(original, str):
        original = np.array(Image.open(original).convert('RGB'), dtype=np.float32)
    if isinstance(compared, str):
        compared = np.array(Image.open(compared).convert('RGB'), dtype=np.float32)

    original = original[None, ...] if original.ndim == 3 else original
    compared = compared[None, ...] if compared.ndim == 3 else compared

    return MultiScaleSSIM(original, compared, max_val=255)


def psnr(original, compared):
    if isinstance(original, str):
        original = np.array(Image.open(original).convert('RGB'), dtype=np.float32)
    if isinstance(compared, str):
        compared = np.array(Image.open(compared).convert('RGB'), dtype=np.float32)

    mse = np.mean(np.square(original - compared))
    psnr = np.clip(
        np.multiply(np.log10(255. * 255. / mse[mse > 0.]), 10.), 0., 99.99)[0]
    return psnr


def latent_rate(feature, mu, sigma):

    gaussian = Normal(mu, sigma)
    pmf = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
    total_bits = torch.sum(torch.clamp(-1.0 * torch.log(pmf + 1e-10) / math.log(2.0), 0, 50))
    
    
    return total_bits



def hyperlatent_rate(z,hyper_cumulative):
        
    def cumulative(x,hyper_cumulative):
        if hyper_cumulative == "sigmoid":
            return torch.sigmoid(x)    
        else:
            half = 0.5
            const = -(2 ** -0.5)
            return half * torch.erf(const * x)
            
 
    pmf = cumulative(z + .5,hyper_cumulative) - cumulative(z - .5,hyper_cumulative)
    total_bits = torch.sum(torch.clamp(-1.0 * torch.log(pmf + 1e-10) / math.log(2.0), 0, 50))
    
    return total_bits


def BD_rate(R1,PSNR1,R2,PSNR2):
  log_R1 = np.log(R1)
  log_R2 = np.log(R2)

  # Fit 3rd-degree polynomials
  p1 = np.polyfit(PSNR1, log_R1, 3)
  p2 = np.polyfit(PSNR2, log_R2, 3)

  # Integration interval (overlap range)
  p_min = max(min(PSNR1), min(PSNR2))
  p_max = min(max(PSNR1), max(PSNR2))

  # Integrate the polynomials
  p_int1 = np.polyint(p1)
  p_int2 = np.polyint(p2)

  # Evaluate integrals and compute difference
  int1 = np.polyval(p_int1, p_max) - np.polyval(p_int1, p_min)
  int2 = np.polyval(p_int2, p_max) - np.polyval(p_int2, p_min)

  # Average difference (exponentiated because log domain)
  avg_diff = (int2 - int1) / (p_max - p_min)
  return (np.exp(avg_diff) - 1) * 100  # BD-rate in %


def mAP(preds,gts,num_classes):
  metric = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=num_classes)

  for pred, gt in zip(preds, gts):
      # convert to numpy if torch tensor
      if torch.is_tensor(pred):
          pred = pred.detach().cpu().numpy()
      if torch.is_tensor(gt):
          gt = gt.detach().cpu().numpy()

      # ensure float64 dtype for metric ops
      pred = np.asarray(pred, dtype=np.float64)
      gt   = np.asarray(gt, dtype=np.float64)

      # if your preds are [x1,y1,x2,y2,score,class] and lib wants [x1,y1,x2,y2,class,score], swap:
      pred = pred[:, [0, 1, 2, 3, 5, 4]]

      metric.add(preds=pred, gt=gt)
  results = metric.value(iou_thresholds=[0.5])

  return results['mAP']


def make_gt(bboxes,class_ids,difficults,crowds):
  batch_gts = []
  for b, c, d, r in zip(bboxes, class_ids, difficults, crowds):
      if b.numel() == 0:  # no boxes for this image
          gt_arr = np.empty((0, 7), dtype=np.float32)
      else:
          b = b.reshape(-1, 4).float().cpu().numpy()
          c = c.reshape(-1, 1).cpu().numpy()
          d = d.reshape(-1, 1).cpu().numpy()
          r = r.reshape(-1, 1).cpu().numpy()
          gt_arr = np.concatenate([b, c, d, r], axis=1)
      batch_gts.append(gt_arr)
  return batch_gts


def pack_yolo_preds_to_tm(preds_list,device):
    """
    preds_list: List[Tensor[Ni, 6]] with [x1,y1,x2,y2,score,class_id]
    returns:    List[dict] for torchmetrics
    """
    out = []
    for p in preds_list:
        if p.numel() == 0:
            out.append({"boxes": torch.empty(0,4,device=device),
                        "scores": torch.empty(0,   device=device),
                        "labels": torch.empty(0, dtype=torch.long, device=device)})
            continue
        out.append({
            "boxes":  p[:, :4].to(device=device, dtype=torch.float32),
            "scores": p[:, 4].to(device=device, dtype=torch.float32),
            "labels": p[:, 5].to(device=device, dtype=torch.long),
        })
    return out
  

def pack_targets_to_tm(bboxes_list, class_ids_list, device=None):
    out = []
    for i, (b, c) in enumerate(zip(bboxes_list, class_ids_list)):
        if b.numel() == 0:
            d = {"boxes": torch.empty(0,4, device=device),
                 "labels": torch.empty(0, dtype=torch.long, device=device)}
        else:
            d = {"boxes": b.to(device=device, dtype=torch.float32),
                 "labels": c.to(device=device, dtype=torch.long)}
        out.append(d)
    return out

if __name__ == "__main__":
    
    original_image = '/home/nguyensolbadguy/Code_Directory/compression/models/yolov3/barbara.bmp'
    compared_image = '/home/nguyensolbadguy/Code_Directory/compression/models/yolov3/compressed_barbara.jpg'
    
    # print(msssim(original_image, compared_image),end=' ')
    # print(psnr(original_image, compared_image),end=' ')
    
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
    
    original = np.array(Image.open(original_image).convert('RGB'), dtype=np.float32)
    compared = np.array(Image.open(compared_image).convert('RGB'), dtype=np.float32)
    
    transform =transforms.Compose([transforms.ToTensor()])
    original_tensor = transform(original).unsqueeze(0)
    compared_tensor = transform(compared).unsqueeze(0)
    
    # print(ms_ssim(original_tensor, compared_tensor))
    
    # test bit rate
    means = torch.zeros([1,128,16,16])
    variances = torch.ones([1,128,16,16])*0.1
    feature = torch.randn([1, 128,16,16])
    
    z = torch.rand([1,192,4,4])
    
    hyperBit,_ = hyperlatent_rate(z,'sigmoid')
    latentBit,_= latent_rate(feature, means, variances)
    
    print(hyperBit)
    print(latentBit)
    
    print((hyperBit + latentBit) / (256*256))
    
    
    # test mAP
    # [xmin, ymin, xmax, ymax, class_id, confidence]
    preds = np.array([
    [50, 50, 150, 150,1, 0.65]
    ])

    # [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
    gts = np.array([
    [50, 50, 150, 150,1, 0, 0]
    ])
    
    map = mAP(preds,gts,num_classes=2)
    
    print(map['mAP'])
    