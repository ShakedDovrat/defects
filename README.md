## Assumptions
1. Registration between reference and inspection images is translation only.
2. There are sufficient interest points in the images in order to find transformation.

## Steps
1. Pre-process - Median filter
2. Registration - Translation and also calc valid mask
3. Diff - absdiff, hysteresis, logical and with valid mask
4. Post-process - opening & closing
    

#### Ideas
1. Abs diff is problematic, should use relative diff
