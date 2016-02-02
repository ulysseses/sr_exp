# test1
python test_paper.py --command train --path_conf paper/test1/a.yaml
python test_paper.py --command infer --path_conf paper/test1/a.yaml 2>&1 | \
    tee -a notes/1a_in.txt

python test_paper.py --command train --path_conf paper/test1/b.yaml
python test_paper.py --command infer --path_conf paper/test1/b.yaml 2>&1 | \
    tee -a notes/1b_in.txt

# test2
python test_paper.py --command train --path_conf paper/test2/a.yaml
python test_paper.py --command infer --path_conf paper/test2/a.yaml 2>&1 | \
    tee -a notes/2a_in.txt

python test_paper.py --command train --path_conf paper/test2/b.yaml
python test_paper.py --command infer --path_conf paper/test2/b.yaml 2>&1 | \
    tee -a notes/2b_in.txt

# test3
python test_paper.py --command train --path_conf paper/test3/a.yaml
python test_paper.py --command infer --path_conf paper/test3/a.yaml 2>&1 | \
    tee -a notes/3a_in.txt

python test_paper.py --command train --path_conf paper/test3/b.yaml
python test_paper.py --command infer --path_conf paper/test3/b.yaml 2>&1 | \
    tee -a notes/3b_in.txt

python test_paper.py --command train --path_conf paper/test3/c.yaml
python test_paper.py --command infer --path_conf paper/test3/c.yaml 2>&1 | \
    tee -a notes/3c_in.txt

# # test4
# python test_paper.py --command train --path_conf paper/test4/a.yaml
# python test_paper.py --command infer --path_conf paper/test4/a.yaml 2>&1 | \
#     tee -a notes/4a_in.txt

# python test_paper.py --command train --path_conf paper/test4/b.yaml
# python test_paper.py --command infer --path_conf paper/test4/b.yaml 2>&1 | \
#     tee -a notes/4b_in.txt

# python test_paper.py --command train --path_conf paper/test4/c.yaml
# python test_paper.py --command infer --path_conf paper/test4/c.yaml 2>&1 | \
#     tee -a notes/4c_in.txt

# test5
python test_paper.py --command train --path_conf paper/test5/a.yaml
python test_paper.py --command infer --path_conf paper/test5/a.yaml 2>&1 | \
    tee -a notes/5a_in.txt

python test_paper.py --command train --path_conf paper/test5/b.yaml
python test_paper.py --command infer --path_conf paper/test5/b.yaml 2>&1 | \
    tee -a notes/5b_in.txt

