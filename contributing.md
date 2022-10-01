

unzip datasets.zip
git filter-branch --tree-filter 'find . -name "*.npy" -delete' HEAD
git lfs prune


rm -rf .git
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:EffiSciencesResearch/hackathon42.git
git push --set-upstream origin main -f
