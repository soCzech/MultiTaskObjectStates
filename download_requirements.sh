echo -n "Downloading and patching clip repository ... "
git clone --quiet https://github.com/openai/CLIP.git || exit 1
cd CLIP || exit 1
git checkout --quiet d50d76daa670286dd6cacf3bcd80b5e4823fc8e1 || exit 1
sed -i /self.proj/d clip/model.py || exit 1
cd ..
echo "OK ✓"

echo -n "Downloading clip weights ... "
mkdir -p weights
wget https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt -q -O weights/ViT-L-14.pt
SHA256SUM=$(sha256sum weights/ViT-L-14.pt | cut -d' ' -f1)

if [[ ${SHA256SUM} == "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836" ]]; then
  echo "OK ✓"
else
  echo "ERROR ✗"
  exit 1
fi

echo -n "Downloading ChangeIt annotations ... "
mkdir -p videos
git clone --quiet https://github.com/soCzech/ChangeIt.git || exit 1
echo "OK ✓"

echo "To replicate our experiments, please download ChangeIt videos into \`videos/*category*\` folders."
echo "More details on how to download the videos at https://github.com/soCzech/ChangeIt."
echo "If you wish to train the model on your data, please see the README file."
