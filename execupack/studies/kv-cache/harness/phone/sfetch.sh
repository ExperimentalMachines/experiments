#!/system/bin/sh
# usage: sfetch.sh URLFILE: each line "WINDOW URL SHA256"; downloads to study/pte/WINDOW/<name> (up to 5 tries) and
# writes <name>.ok only when the file's sha256 matches, which is what study.sh waits for. A lock directory per file
# keeps two fetches from ever writing the same file (that corrupted files once, 2026-11-20). Runs detached.
S=/data/local/tmp/kvq/study
while read w url sum; do
  [ -z "$url" ] && continue
  f=$S/pte/$w/${url##*/}
  mkdir -p $S/pte/$w
  [ -f "$f.ok" ] && continue
  mkdir "$f.lock" 2>/dev/null || continue
  (
    for i in 1 2 3 4 5; do
      if curl -sS -f -o "$f" "$url" && [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$sum" ]; then touch "$f.ok"; break; fi
      echo "$(date +%H:%M:%S) retry $i ${url##*/}" >> $S/out/progress.log
      sleep 10
    done
    rmdir "$f.lock"
  ) &
done < "$1"
wait
echo "$(date +%H:%M:%S) fetched $1" >> $S/out/progress.log
