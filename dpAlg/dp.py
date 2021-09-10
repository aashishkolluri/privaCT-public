import numpy as np

# Construct noisy degree vectors w.r.t partitions
def histogramLDP(hist, eps_h, prng=np.random):
    hist = hist.astype(np.float32)
    out_hist = np.empty(len(hist), dtype=np.float32)
    # individual count query for each bucket
    for i in range(len(hist)):
        out_hist[i] = countLDP(hist[i], eps_h, prng=prng)
    return out_hist


"""
    Un comment the below lines to use the function that is implemented better w.r.t floating points.
    We do not use them since we cannot fix the seeds used by the library.
"""
# Taken from the Open source library from Harvard
# https://github.com/opendifferentialprivacy/
# import opendp.whitenoise.core as wn

# # Return noisy counts
# def countLDP(c, eps_c, lower=0, upper=100):
#     with wn.Analysis() as analysis:
#         data = wn.Dataset(value=np.array([1]*c), num_columns=1)
#         # dp_count function by default uses geometric mechanism
#         out_hist = wn.dp_count(
#             data,
#             lower=lower,
#             upper = upper,
#             privacy_usage = {'epsilon': eps_c}
#         )
#     analysis.release()

#     if out_hist.value==None:
#         return c
#     return out_hist.value


# Return noisy counts
def countLDP(c, eps_c, sensitivity=1.0, prng=np.random):
    """
        Does not handle the privacy leaks due to floating point precision
    """
    if eps_c == 0.0:
        return c
    return c + prng.laplace(0, sensitivity / eps_c)
