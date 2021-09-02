mkdir model
mkdir model/la_with_lmbd_8
file_id="1qlvPquVLfP20neHYZ6cwzmzAw9s-O0Yf"
file_name="./model/la_with_lmbd_8/net_best.pth"
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${file_id}" -o ${file_name}