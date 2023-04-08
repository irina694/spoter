search_dir=./train
for entry in "$search_dir"/*
do
  videoid="$(basename $entry)"
  echo "$search_dir/$videoid.gif"
  convert -delay 15 "$entry"/* "$search_dir/$videoid.gif"
done
