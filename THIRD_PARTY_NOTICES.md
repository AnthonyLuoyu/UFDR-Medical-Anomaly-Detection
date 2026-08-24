# Third-Party Notices

This file records third-party code adapted by this minimal UFDR package. It is not a project-level license for the remainder of the package.

## PUCL contrastive objective

Parts of the uncertainty-aware contrastive objective in `ufdr/pucl.py` were adapted from the local upstream source tree `extra_network/SpatialCL-master`. That source is distributed under the following MIT License; the text below is copied verbatim from its `LICENSE` file.

```text
MIT License

Copyright (c) 2025 Olemou Felix

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## External DINOv3 assets and provider

DINOv3 weights and the `lightly_train` provider are not bundled with this package. Users must obtain them separately and comply with each upstream project's license and terms. The MIT notice above applies only to the identified adapted contrastive-objective code; it does not relicense DINOv3, its weights, or the provider.

## Library dependencies

This package calls PyTorch and other libraries as installed dependencies; it does not copy their source code. In particular, the included decoder and RCA implementations do not import or copy torchvision APIs or source. Each installed dependency remains subject to its own upstream license.
