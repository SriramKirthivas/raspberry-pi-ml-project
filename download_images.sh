#!/bin/sh

csv_file="$1"
shortname="$2"
output_dir="images"

mkdir -p "$output_dir"

# Read all filenames from CSV into an array
filenames=$(cat "$csv_file")

total=0
already_present=0
to_download=0

# Level 1 stats: Check which files are already present
for line in $filenames; do
    total=$((total + 1))
    filename="${line%?}"
    filepath="${output_dir}/${filename}"
    if [ -f "$filepath" ]; then
        already_present=$((already_present + 1))
    else
        to_download=$((to_download + 1))
    fi
done

echo "Level 1 Stats:"
echo "--------------"
echo "Total images in CSV: $total"
echo "Already present: $already_present"
echo "To download: $to_download"
echo ""

downloaded=0
errors=0

start_time=$(date +%s)

# Download missing files
for line in $filenames; do
    filename="${line%?}"
    filepath="${output_dir}/${filename}"

    if [ -f "$filepath" ]; then
        continue
    fi

    curl --location --request GET "cs7ns1.scss.tcd.ie?shortname=${shortname}&myfilename=${filename}" -o "$filepath"
    if [ $? -eq 0 ] && [ -s "$filepath" ]; then
        downloaded=$((downloaded + 1))
    else
        errors=$((errors + 1))
        rm -f "$filepath"
    fi
done

# Level 2 stats: Check for missing files
missing=0
for line in $filenames; do
    filename="${line%?}"
    filepath="${output_dir}/${filename}"
    if [ ! -f "$filepath" ]; then
        missing=$((missing + 1))
    fi
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo ""
echo "Level 2 Stats:"
echo "--------------"
echo "Total images in CSV: $total"
echo "Downloaded this run: $downloaded"
echo "Errors: $errors"
echo "Missing images: $missing"
echo "Elapsed time: ${elapsed} seconds"