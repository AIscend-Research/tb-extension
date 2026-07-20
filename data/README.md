# data/

Put the cohort images here. Nothing in this folder except this README is
committed (see `.gitignore`) so raw images and manifests never end up in git.

Expected layout (only the cohorts you have need to exist):

    data/
      montgomery/CXR_png/MCUCXR_####_X.png
      shenzhen/CXR_png/CHNCXR_####_X.png
      niaid/labels.csv                 # columns: image_path,label
      niaid/<images...>
      rsna/stage_2_train_labels.csv
      rsna/stage_2_train_images/*.dcm
      manifest.csv                     # written by scripts/build_manifest.py

See ../docs/DATA.md for download links, licences, and the label conventions.
