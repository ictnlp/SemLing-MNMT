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

# decoding and computing test bleu on top 3 checkpoint

model=$setting
out_file=$output_dir/bleu_test

# rm $out_file
# rm -r $output_dir/out/$model
mkdir -p $output_dir/out/$model
mkdir -p $output_dir/data/reference

ref_dir=$output_dir/data/reference
prefix=test

deen_en=$ref_dir/${prefix}.de-en.en
ende_de=$ref_dir/${prefix}.en-de.de
enit_it=$ref_dir/${prefix}.en-it.it
ennl_nl=$ref_dir/${prefix}.en-nl.nl
enro_ro=$ref_dir/${prefix}.en-ro.ro
iten_en=$ref_dir/${prefix}.it-en.en
nlen_en=$ref_dir/${prefix}.nl-en.en
roen_en=$ref_dir/${prefix}.ro-en.en

for file in $checkpoints/*
do
    echo $file >> $out_file
    gen_sen 0 $file de en $model test  &
    gen_sen 1 $file en de $model test  &
    gen_sen 2 $file en it $model test  &
    gen_sen 3 $file en nl $model test  &
    # wait
    gen_sen 4 $file en ro $model test  &
    gen_sen 5 $file it en $model test  &
    gen_sen 6 $file nl en $model test  &
    gen_sen 7 $file ro en $model test  &
    wait

    declare -a bleu_enxx
    bleu_enxx[0]=en-de`cat $output_dir/out/$model/en-de.$(basename $file).test | sacrebleu $ende_de -w 2`
    bleu_enxx[1]=en-it`cat $output_dir/out/$model/en-it.$(basename $file).test | sacrebleu $enit_it -w 2`
    bleu_enxx[2]=en-nl`cat $output_dir/out/$model/en-nl.$(basename $file).test | sacrebleu $ennl_nl -w 2`
    bleu_enxx[3]=en-ro`cat $output_dir/out/$model/en-ro.$(basename $file).test | sacrebleu $enro_ro -w 2`

    declare -a b
    sum=0.0
    for ((i=0;i<${#bleu_enxx[*]};i++))
    do
        echo "${bleu_enxx[$i]}" >> $out_file
        b[$i]=`echo ${bleu_enxx[$i]}|sed "s/.*1.5.1\ =\ \([0-9.]\{1,\}\).*/\1/"`
        sum=`echo "scale=2;$sum+${b[$i]}"|bc`
    done
    avg=`echo "scale=2;$sum/${#b[*]}"|bc`
    echo  "En-xx AVG  $avg" >> $out_file

    declare -a bleu_xxen
    bleu_xxen[0]=de-en`cat $output_dir/out/$model/de-en.$(basename $file).test | sacrebleu $deen_en -w 2`
    bleu_xxen[1]=it-en`cat $output_dir/out/$model/it-en.$(basename $file).test | sacrebleu $iten_en -w 2`
    bleu_xxen[2]=nl-en`cat $output_dir/out/$model/nl-en.$(basename $file).test | sacrebleu $nlen_en -w 2`
    bleu_xxen[3]=ro-en`cat $output_dir/out/$model/ro-en.$(basename $file).test | sacrebleu $roen_en -w 2`

    declare -a b1
    sum1=0.0
    for ((i=0;i<${#bleu_xxen[*]};i++))
    do
        echo "${bleu_xxen[$i]}" >> $out_file
        b1[$i]=`echo ${bleu_xxen[$i]}|sed "s/.*1.5.1\ =\ \([0-9.]\{1,\}\).*/\1/"`
        sum1=`echo "scale=2;$sum1+${b1[$i]}"|bc`
    done
    avg1=`echo "scale=2;$sum1/${#b1[*]}"|bc`
    echo  "xx-En AVG  $avg1" >> $out_file

    sum2=`echo "scale=2;$sum+$sum1"|bc`
    avg2=`echo "scale=2;$sum2/(${#b[*]}+${#b1[*]})"|bc`
    echo  "AVG  $avg2" >> $out_file
    
done
