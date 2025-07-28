#!/bin/bash

# Create the new target directory
mkdir -p embedding

# Group_01_audio_1_cleaned_clip1
# Loop through each .npy file in the Utterance_ directory
for filepath in embedding_/Group_*_audio_*_cleaned_clip*.npy; do
    filename=$(basename "$filepath")

    # Extract the IDs using regex
    if [[ $filename =~ Group_([0-9]+)_audio_([0-9]+)_cleaned_clip([0-9]+)\.npy ]]; then
        group_id="${BASH_REMATCH[1]}"
        person_id="${BASH_REMATCH[2]}"
        clip_id="${BASH_REMATCH[3]}"

        # Increment clip_id by 1
        # new_clip_id=$((clip_id + 1))

        # Create the new filename
        new_filename="Group${group_id}_Person${person_id}_Clip${clip_id}.npy"

        # Copy the file to the new directory with the new name
        cp "$filepath" "embedding/$new_filename"
    else
        echo "Skipping unrecognized file format: $filename"
    fi
done

