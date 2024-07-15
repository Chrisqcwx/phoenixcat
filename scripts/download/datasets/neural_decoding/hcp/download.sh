subjects=$(dirname "$(readlink -f "$0")")/subjects.txt

while read -r subj;
do
    echo $subj
done < $subjects