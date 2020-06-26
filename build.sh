source ~/.bashrc
source ~/.bash_profile

path=$(pwd)
static_path="${path}/../lomyal.github.io"
parts=(2016 about archive index.html)

jekyll build
if [ $? -ne 0 ]; then
    echo jekyll build failed
    exit 1
fi

for part in ${parts[@]}; do
    echo ${part}
    rm -rf "${static_path}/${part}"
    cp -r "${path}/_site/${part}" "${static_path}/${part}"
done

