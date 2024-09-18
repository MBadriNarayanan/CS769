# Step 0. Change this to your campus ID
CAMPUSID="9085915255"
mkdir -p $CAMPUSID

# Step 1. (Optional) Any preprocessing step, e.g., downloading pre-trained word embeddings
# wget http://nlp.stanford.edu/data/glove.42B.300d.zip
# unzip glove.42B.300d.zip

# wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
# unzip crawl-300d-2M.vec.zip

# Step 2. Train models on two datasets.
##  2.1. Run experiments on SST
PREF='sst'
python3.8 main.py \
    --train "data/${PREF}-train.txt" \
    --dev "data/${PREF}-dev.txt" \
    --test "data/${PREF}-test.txt" \
    --dev_output "${CAMPUSID}/${PREF}-dev-output.txt" \
    --test_output "${CAMPUSID}/${PREF}-test-output.txt" \
    --model "${CAMPUSID}/${PREF}-model.pt" \
    --word_drop 0.0 \
    --emb_drop 0.2 \
    --hid_drop 0.2 \
    --emb_file "glove.42B.300d.txt"

##  2.2 Run experiments on CF-IMDB
PREF='cfimdb'
python3.8 main.py \
    --train "data/${PREF}-train.txt" \
    --dev "data/${PREF}-dev.txt" \
    --test "data/${PREF}-test.txt" \
    --dev_output "${CAMPUSID}/${PREF}-dev-output.txt" \
    --test_output "${CAMPUSID}/${PREF}-test-output.txt" \
    --model "${CAMPUSID}/${PREF}-model.pt" \
    --word_drop 0.0 \
    --emb_drop 0.2 \
    --hid_drop 0.2 \
    --emb_file "crawl-300d-2M.vec"


# Step 3. Prepare submission:
##  3.1. Copy your code to the $CAMPUSID folder
for file in 'main.py' 'model.py' 'vocab.py' 'setup.py' 'run_exp.sh'; do
	cp $file ${CAMPUSID}/
done
##  3.2. Compress the $CAMPUSID folder to $CAMPUSID.zip (containing only .py/.txt/.pdf/.sh files)
python3.8 prepare_submit.py ${CAMPUSID} ${CAMPUSID}
##  3.3. Submit the zip file to Canvas (https://canvas.wisc.edu/courses/292771/assignments)! Congrats!