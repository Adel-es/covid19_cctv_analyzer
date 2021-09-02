mkdir data
file_id="11Zuc30y2yitxLHScRjTklJZxmOBebl6u"
file_name="./data/Market_pytorch.zip"
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${file_id}" -o ${file_name}
unzip ./data/Market_pytorch.zip -d ./data/