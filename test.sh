echo "this is the first argument: $1"
echo "these are all the arguments $@"

while getopts "l:" arg; do
    if [[ $arg == "l" ]]; then
        echo "arg l was passed with value: ${OPTARG}"
    else
        echo "l was not passed."
    fi
done
