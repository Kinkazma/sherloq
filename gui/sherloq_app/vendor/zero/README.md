ZERO: a local JPEG grid origin detector based on the number of DCT zeros
and its applications in image forensics
========================================================================

Version 4 - Nov 2021

by Tina Nikoukhah <tinanikoukhah@gmail.com>
and Jérémy Anger <jeremy.anger@ens-paris-saclay.fr>
and Rafael Grompone von Gioi <grompone@gmail.com>


Introduction
------------

ZERO is an implementation of the JPEG grid detector applied to forgery
detection in images described in the paper:

     "JPEG Grid Detection based on the Number of DCT Zeros and its
     Application to Automatic and Localized Forgery Detection" by Tina
     Nikoukhah, Jérémy Anger, Thibaud Ehret, Miguel Colom, Jean-Michel
     Morel and Rafael Grompone von Gioi, CVPRW 2019.
[PDF](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Media%20Forensics/Nikoukhah_JPEG_Grid_Detection_based_on_the_Number_of_DCT_Zeros_CVPRW_2019_paper.pdf)

This code is part of the following IPOL publication:

     "ZERO: a local JPEG grid origin detector based on the number of DCT zeros
     and its applications in image forensics" by Tina Nikoukhah, Jérémy Anger,
     Miguel Colom, Jean-Michel Morel, Rafael Grompone von Gioi,
     Image Processing On Line, 2021.
     https://doi.org/10.5201/ipol.2021.390


Online Demo
------------

[IPOL](https://ipolcore.ipol.im/demo/clientApp/demo.html?id=390)


Files
-----

- src/main.c: Main code.

- src/zero.{c,h}: ZERO functions for the libzero library.

- README.md: this file.

- LICENSE: GNU AFFERO GENERAL PUBLIC LICENSE Version 3.

- Makefile: Compilation instructions.

- src/iio.{c,h}: [iio](https://github.com/mnhrdt/iio) code and header.

- visu/requirements.txt: Requirements for python scripts.

- visu/create_votemap.py: Creates a colored vote map.
```
python create_votemap.py votes.png
```

- visu/merge_zero.py: Creates a final visual result which merges the two forgery masks.
```
python merge_zero.py mask_f.png mask_m.png luminance.png
```

- *.{png,jpg}: Test images.


Compiling
---------
The compiling instruction is just
```bash
    make
```
or if you want the code to be parallel
```bash
   make openmp
```
from the directory where the source codes and the Makefile are located.

To compile the library to use the python binding do
```bash
    make libzero
```

To create the compressed version of the image, use imagemagick
```bash
    convert -quality 99% <image> <image_compressed.jpg>
```

To verify a correct compilation you can apply the algorithm to the
test images. This can be done by executing:
```bash
    make test
```

This should print the following message:
```
test on roma.png
----------------
./zero roma.png roma99.jpg
No overall JPEG grid found.

No suspicious traces found in the image with the performed analysis.

test on pelican.png
-------------------
./zero pelican.png pelican99.jpg
main grid found: #6 (6,0) log(nfa) = -6373.72

The most meaningful JPEG grid origin is not (0,0).
This may indicate that the image has been cropped.

test on tampered1.png
---------------------
./zero tampered1.png tampered1_99.jpg
No overall JPEG grid found.

A meaningful grid was found here:
bounding box: 104 94 to 153 159 [50x66] grid: #0 (0,0) log(nfa) = -25.8163

Suspicious traces found in the image.
This may be caused by image manipulations such as resampling,
copy-paste, splicing.  Please examine the deviant meaningful region
to make your own opinion about a potential forgery.

test on tampered2.png
---------------------
./zero tampered2.png tampered2_99.jpg
main grid found: #6 (6,0) log(nfa) = -6188.44

A meaningful grid different from the main one was found here:
bounding box: 330 68 to 401 111 [72x44] grid: #34 (2,4) log(nfa) = -39.2402

The most meaningful JPEG grid origin is not (0,0).
This may indicate that the image has been cropped.

Suspicious traces found in the image.
This may be caused by image manipulations such as resampling,
copy-paste, splicing.  Please examine the deviant meaningful region
to make your own opinion about a potential forgery.
```

Running
-------

The program is executed as:
```
    ./zero
```

That should print the following usage:
```
    error: usage: zero <image> [image_jpeg99]
    finds JPEG grids and forgeries
```

The command takes an input image as argument. Any image file format handled by
Enric Meinhardt-Llopis' IIO library can be used. This includes, for example,
PNG, JPG, TIFF, PPM, PGM.

A typical execution is as follows:
```
    ./zero pelican.png pelican99.jpg
```

To get a nice visual result, the python codes in visu/ can be used as follows:
```
    python visu/create_votemap.py votes.png
    python visu/create_votemap.py votes_jpeg.png
    python visu/merge_zero.py mask_f.png mask_m.png luminance.png
```
to get colored vote maps and a final colored result for the forgery detection. 


Outputs
-------
- luminance.png: Luminance image.
- votes.png: Vote map.
- votes_jpeg.png: Altered vote map of the compressed version.
- mask_f.png: Foreign mask F of foreign grids.
- mask_m.png: Forgery mask M of missing grids.
- colored_votes.png: Colored vote map.
- colored_votes_jpeg.png: Colored altered vote map of the compressed version.
- result_zero.png : Forgery detection result, in blue the result from mask M,
  in red the result from mask F.

For example, the line:
```
    main grid found: #6 (6,0) log(nfa) = -6373.72

    The most meaningful JPEG grid origin is not (0,0).
    This may indicate that the image has been cropped.
```
means that the main grid is (6,0) with a NFA at 10^-6373.72.


Copyright and License
---------------------

Copyright (c) 2018-2021 Tina Nikoukhah <tinanikoukhah@gmail.com>
Copyright (c) 2018-2021 Jérémy Anger <jeremy.anger@ens-paris-saclay.fr>
Copyright (c) 2018-2021 Rafael Grompone von Gioi <grompone@gmail.com>


ZERO is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

ZERO is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.


Thanks
------

We would be grateful to receive any comment, especially about errors,
bugs, or strange results.
