

unzip datasets.zip
git filter-branch --tree-filter 'find . -name "*.npy" -delete' HEAD
git lfs prune

