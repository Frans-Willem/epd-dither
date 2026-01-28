epd-dither
==========
This library implements methods used to dither images for display on 6-color e-paper displays such as the Spectra 6.

Differences to other libraries
------------------------------
This library differs from most other libraries (such as, for example, [epdoptimize](https://github.com/Utzel-Butzel/epdoptimize)) in that it doesn't dither directly using RGB values, but first attempts to decompose each pixel into a mix of possible colors (e.g. gray would be decomposed to 50% white + 50% black) and using those values to pick a dithered color. In my experience this leads to better results in colors that are outside the range of colors that rae possible to accurately create from the e-paper's base colors.
