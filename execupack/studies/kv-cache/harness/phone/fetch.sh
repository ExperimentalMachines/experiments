#!/system/bin/sh
# usage: fetch.sh URLFILE MARKER: downloads every URL into pte/ in parallel (retried), unpacks any .tar into the
# working dir, then touches MARKER. Runs detached on the phone.
D=/data/local/tmp/kvq
cd $D || exit 1
mkdir -p pte out
while read url; do
  [ -z "$url" ] && continue
  f=pte/${url##*/}
  [ -f "$f.ok" ] && continue
  ( for i in 1 2 3 4 5; do curl -sS -f -o "$f" "$url" && touch "$f.ok" && break; sleep 10; done ) &
done < "$1"
wait
for t in pte/*.tar; do [ -f "$t" ] && tar xf "$t" -C $D && rm "$t"; done
echo "$(date +%H:%M:%S) fetched $1: $(ls pte/*.ok 2>/dev/null | wc -l) files ok" >> out/progress.log
touch "$2"
