# test1
for i in {1..3}
do
    python test_paper.py --command train --path_conf paper/test1/a.yaml 2>&1 | \
        tee -a tmp/paper/test1/a/log_tr.txt
    python test_paper.py --command infer --path_conf paper/test1/a.yaml 2>&1 | \
        tee -a tmp/paper/test1/a/log_in.txt
done

for i in {1..3}
do
    python test_paper.py --command train --path_conf paper/test1/b.yaml 2>&1 | \
        tee -a tmp/paper/test1/b/log_tr.txt
    python test_paper.py --command infer --path_conf paper/test1/b.yaml 2>&1 | \
        tee -a tmp/paper/test1/b/log_in.txt
done

# test2
for i in {1..3}
do
    python test_paper.py --command train --path_conf paper/test2/a.yaml 2>&1 | \
        tee -a tmp/paper/test2/a/log_tr.txt
    python test_paper.py --command infer --path_conf paper/test2/a.yaml 2>&1 | \
        tee -a tmp/paper/test2/a/log_in.txt
done

for i in {1..3}
do
    python test_paper.py --command train --path_conf paper/test2/b.yaml 2>&1 | \
        tee -a tmp/paper/test2/b/log_tr.txt
    python test_paper.py --command infer --path_conf paper/test2/b.yaml 2>&1 | \
        tee -a tmp/paper/test2/b/log_in.txt
done

# test3
for i in {1..3}
do
    python test_paper.py --command train --path_conf paper/test3/a.yaml 2>&1 | \
        tee -a tmp/paper/test3/a/log_tr.txt
    python test_paper.py --command infer --path_conf paper/test3/a.yaml 2>&1 | \
        tee -a tmp/paper/test3/a/log_in.txt
done

for i in {1..3}
do
    python test_paper.py --command train --path_conf paper/test3/b.yaml 2>&1 | \
        tee -a tmp/paper/test3/b/log_tr.txt
    python test_paper.py --command infer --path_conf paper/test3/b.yaml 2>&1 | \
        tee -a tmp/paper/test3/b/log_in.txt
done

for i in {1..3}
do
    python test_paper.py --command train --path_conf paper/test3/c.yaml 2>&1 | \
        tee -a tmp/paper/test3/c/log_tr.txt
    python test_paper.py --command infer --path_conf paper/test3/c.yaml 2>&1 | \
        tee -a tmp/paper/test3/c/log_in.txt
done

# test4
for i in {1..3}
do
    python test_paper.py --command train --path_conf paper/test4/a.yaml 2>&1 | \
        tee -a tmp/paper/test4/a/log_tr.txt
    python test_paper.py --command infer --path_conf paper/test4/a.yaml 2>&1 | \
        tee -a tmp/paper/test4/a/log_in.txt
done

for i in {1..3}
do
    python test_paper.py --command train --path_conf paper/test4/b.yaml 2>&1 | \
        tee -a tmp/paper/test4/b/log_tr.txt
    python test_paper.py --command infer --path_conf paper/test4/b.yaml 2>&1 | \
        tee -a tmp/paper/test4/b/log_in.txt
done

for i in {1..3}
do
    python test_paper.py --command train --path_conf paper/test4/c.yaml 2>&1 | \
        tee -a tmp/paper/test4/c/log_tr.txt
    python test_paper.py --command infer --path_conf paper/test4/c.yaml 2>&1 | \
        tee -a tmp/paper/test4/c/log_in.txt
done

# test5
for i in {1..3}
do
    python test_paper.py --command train --path_conf paper/test5/a.yaml 2>&1 | \
        tee -a tmp/paper/test5/a/log_tr.txt
    python test_paper.py --command infer --path_conf paper/test5/a.yaml 2>&1 | \
        tee -a tmp/paper/test5/a/log_in.txt
done

for i in {1..3}
do
    python test_paper.py --command train --path_conf paper/test5/b.yaml 2>&1 | \
        tee -a tmp/paper/test5/b/log_tr.txt
    python test_paper.py --command infer --path_conf paper/test5/b.yaml 2>&1 | \
        tee -a tmp/paper/test5/b/log_in.txt
done

