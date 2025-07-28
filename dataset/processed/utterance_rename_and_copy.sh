#!/bin/bash

# Create the new target directory
mkdir -p Utterance

# Loop through each .npy file in the Utterance_ directory
for filepath in Utterance_/Group_*_Person_*_Clip*.npy; do
    filename=$(basename "$filepath")

    # Extract the IDs using regex
    if [[ $filename =~ Group_([0-9]+)_Person_([0-9]+)_Clip([0-9]+)\.npy ]]; then
        group_id="${BASH_REMATCH[1]}"
        person_id="${BASH_REMATCH[2]}"
        clip_id="${BASH_REMATCH[3]}"

        # Increment clip_id by 1
        new_clip_id=$((clip_id + 1))

        # Create the new filename
        new_filename="Group${group_id}_Person${person_id}_Clip${new_clip_id}.npy"

        # Copy the file to the new directory with the new name
        cp "$filepath" "Utterance/$new_filename"
    else
        echo "Skipping unrecognized file format: $filename"
    fi
done

