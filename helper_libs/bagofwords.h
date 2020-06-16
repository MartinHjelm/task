#ifndef BAGOFWORDS_H
#define BAGOFWORDS_H

// std
#include <vector>

// Eigen
#include <Eigen/Dense>


class BagOfWords
{
  int codeBookSize_;
  Eigen::MatrixXd codeBook_;

public:
  BagOfWords();
  BagOfWords(const int &size);

  // Computes the code book from the given data
  double computeCodeBook(const Eigen::MatrixXd &X, const int &kmIters=100, const int &kmReTrials=1);

  // Looks up the codebook code word for a given row vector
  int lookUpCodeWord(const Eigen::VectorXd &vec);

  // Looks up the codebook codes for a given row vector matrix
  std::vector<int> lookUpCodeWords(const Eigen::MatrixXd &X);

  inline Eigen::MatrixXd getCodeBook() const { return codeBook_; }
  inline void setCodeBook(Eigen::MatrixXd cb) { codeBook_=cb; codeBookSize_=cb.rows(); }
  inline void setCodeBookSize(int size) { codeBookSize_ = size; }
  inline int codeBookSize() const { return codeBookSize_; }
};

#endif // BAGOFWORDS_H
