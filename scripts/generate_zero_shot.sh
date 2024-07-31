checkpoints=/path/to/your/checkpoints
setting=/your/setting
lang_pairs='de-en,en-de,en-it,en-nl,en-ro,it-en,nl-en,ro-en'
langs='de,en,it,nl,ro'
datapath=/path/to/your/test/data
output_dir=/path/to/your/output_dir
choose_translatoin=./tools/choose-translation.py

#gpu=1
#model=2
#src_lang=3
#tgt_lang=4
#model file=5
#valid/test=6

gen_sen(){
    CUDA_VISIBLE_DEVICES=$1 fairseq-generate \
        $datapath \
        --fp16 \
        --path $2 \
        --gen-subset $6 \
        --beam 5 \
        --max-tokens 1024 \
        --remove-bpe 'sentencepiece' \
        --lenpen 1 \
        --task translation_multi_simple_epoch \
        --lang-pairs $lang_pairs \
        --langs $langs \
        --source-lang $3 \
        --target-lang $4  \
        --encoder-langtok src \
        --decoder-langtok \
        --left-pad-source False --left-pad-target False \
        --dataset-impl mmap \
        --skip-invalid-size-inputs-valid-test > $output_dir/out/$5/gen.out.$(basename $2).$3-$4.$6
    python $choose_translatoin $output_dir/out/$5/gen.out.$(basename $2).$3-$4.$6 $output_dir/out/$5/$3-$4.$(basename $2).$6
    
    if [ ! -f $output_dir/data/reference/$6.$3-$4.$4 ]; then
        grep ^T $output_dir/out/$5/gen.out.$(basename $2).$3-$4.$6 | sort -n -k 2 -t '-' | cut -f 2 > $output_dir/data/reference/$6.$3-$4.$4
    fi
}

# decoding and computing zero-shot bleu

model=$setting
out_file=$output_dir/bleu_zero-shot

# rm $out_file
# rm -r $output_dir/out/$model
mkdir -p $output_dir/out/$model
mkdir -p $output_dir/data/reference

ref_dir=$output_dir/data/reference
prefix=test

deit_it=$ref_dir/${prefix}.de-it.it
denl_nl=$ref_dir/${prefix}.de-nl.nl
dero_ro=$ref_dir/${prefix}.de-ro.ro
itde_de=$ref_dir/${prefix}.it-de.de
itnl_nl=$ref_dir/${prefix}.it-nl.nl
itro_ro=$ref_dir/${prefix}.it-ro.ro
nlde_de=$ref_dir/${prefix}.nl-de.de
nlit_it=$ref_dir/${prefix}.nl-it.it
nlro_ro=$ref_dir/${prefix}.nl-ro.ro
rode_de=$ref_dir/${prefix}.ro-de.de
roit_it=$ref_dir/${prefix}.ro-it.it
ronl_nl=$ref_dir/${prefix}.ro-nl.nl

for file in $checkpoints/*
do

    echo $file >> $out_file
    gen_sen 0 $file de it $model test  &
    gen_sen 1 $file de nl $model test  &
    gen_sen 2 $file de ro $model test  &
    gen_sen 3 $file it de $model test  &
    # wait
    gen_sen 4 $file it nl $model test  &
    gen_sen 5 $file it ro $model test  &
    gen_sen 6 $file nl de $model test  &
    gen_sen 7 $file nl it $model test  &
    # wait
    gen_sen 0 $file nl ro $model test  &
    gen_sen 1 $file ro de $model test  &
    gen_sen 2 $file ro it $model test  &
    gen_sen 3 $file ro nl $model test  &
    wait

    declare -a bleu
    bleu[0]=de-it`cat $output_dir/out/$model/de-it.$(basename $file).test | sacrebleu $deit_it -w 2`
    bleu[1]=de-nl`cat $output_dir/out/$model/de-nl.$(basename $file).test | sacrebleu $denl_nl -w 2`
    bleu[2]=de-ro`cat $output_dir/out/$model/de-ro.$(basename $file).test | sacrebleu $dero_ro -w 2`
    bleu[3]=it-de`cat $output_dir/out/$model/it-de.$(basename $file).test | sacrebleu $itde_de -w 2`
    bleu[4]=it-nl`cat $output_dir/out/$model/it-nl.$(basename $file).test | sacrebleu $itnl_nl -w 2`
    bleu[5]=it-ro`cat $output_dir/out/$model/it-ro.$(basename $file).test | sacrebleu $itro_ro -w 2`
    bleu[6]=nl-de`cat $output_dir/out/$model/nl-de.$(basename $file).test | sacrebleu $nlde_de -w 2`
    bleu[7]=nl-it`cat $output_dir/out/$model/nl-it.$(basename $file).test | sacrebleu $nlit_it -w 2`
    bleu[8]=nl-ro`cat $output_dir/out/$model/nl-ro.$(basename $file).test | sacrebleu $nlro_ro -w 2`
    bleu[9]=ro-de`cat $output_dir/out/$model/ro-de.$(basename $file).test | sacrebleu $rode_de -w 2`
    bleu[10]=ro-it`cat $output_dir/out/$model/ro-it.$(basename $file).test | sacrebleu $roit_it -w 2`
    bleu[11]=ro-nl`cat $output_dir/out/$model/ro-nl.$(basename $file).test | sacrebleu $ronl_nl -w 2`

    sum=0.0
    # declare -a b1
    for ((i=0;i<${#bleu[*]};i++)); do
    echo "${bleu[$i]}" >> $out_file
        b1[$i]=`echo ${bleu[$i]}|sed -e "s/.*1.5.1\ =\ \([0-9.]\{1,\}\).*/\1/"`
        # echo "this is b1 ${b1[$i]}"
        sum=`echo "scale=2;$sum+${b1[$i]}"|bc`
    done
    avg=`echo "scale=2;$sum/${#bleu[*]}"|bc`
    echo "AVG  $avg" >> $out_file
    
done