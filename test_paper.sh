# # test1
# python test_paper.py --command train --path_conf paper/test1/a.yaml
# python test_paper.py --command infer --path_conf paper/test1/a.yaml 2>&1 | \
#     tee notes/1a_in.txt

# python test_paper.py --command train --path_conf paper/test1/b.yaml
# python test_paper.py --command infer --path_conf paper/test1/b.yaml 2>&1 | \
#     tee notes/1b_in.txt

# # test2
# python test_paper.py --command train --path_conf paper/test2/a.yaml
# python test_paper.py --command infer --path_conf paper/test2/a.yaml 2>&1 | \
#     tee notes/2a_in.txt

# python test_paper.py --command train --path_conf paper/test2/b.yaml
# python test_paper.py --command infer --path_conf paper/test2/b.yaml 2>&1 | \
#     tee notes/2b_in.txt

# # test3
# python test_paper.py --command train --path_conf paper/test3/a.yaml
# python test_paper.py --command infer --path_conf paper/test3/a.yaml 2>&1 | \
#     tee notes/3a_in.txt

# python test_paper.py --command train --path_conf paper/test3/b.yaml
# python test_paper.py --command infer --path_conf paper/test3/b.yaml 2>&1 | \
#     tee notes/3b_in.txt

# python test_paper.py --command train --path_conf paper/test3/c.yaml
# python test_paper.py --command infer --path_conf paper/test3/c.yaml 2>&1 | \
#     tee notes/3c_in.txt

# # test5
# python test_paper.py --command train --path_conf paper/test5/a.yaml
# python test_paper.py --command infer --path_conf paper/test5/a.yaml 2>&1 | \
#     tee notes/5a_in.txt

# python test_paper.py --command train --path_conf paper/test5/b.yaml
# python test_paper.py --command infer --path_conf paper/test5/b.yaml 2>&1 | \
#     tee notes/5b_in.txt

# # test 6
# python test_paper.py --command train --path_conf paper/test6/a.yaml
# python test_paper.py --command infer --path_conf paper/test6/a.yaml 2>&1 | \
#     tee notes/6a_in.txt
    
python test_paper.py --command train --path_conf paper/test6/b.yaml
python test_paper.py --command infer --path_conf paper/test6/b.yaml 2>&1 | \
    tee notes/6b_in.txt