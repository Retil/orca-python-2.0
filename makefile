LIBSVM_DIR = classifiers/libsvmRank/python
SVOREX_DIR = classifiers/svorex
SVMOP_DIR = classifiers/SVMOP/python

.PHONY: clean subdirs

subdirs: $(LIBSVM_DIR) $(SVOREX_DIR) $(SVMOP_DIR)
	$(MAKE) -e -C $(LIBSVM_DIR)
	$(MAKE) -e -C $(SVOREX_DIR)
	$(MAKE) -e -C $(SVMOP_DIR)
	
clean: $(LIBSVM_DIR) $(SVOREX_DIR)
	$(MAKE) -e -C $(LIBSVM_DIR) clean
	$(MAKE) -e -C $(SVOREX_DIR) clean
	$(MAKE) -e -C $(SVMOP_DIR) clean

