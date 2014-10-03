#include<map>
#include<fstream>

/**
 *
 * \fn GetMatrixMarket_CSR_sym
 *  put matrix saved in MatrixMarket format onto vectors representing CSR  
 *
 */

template<typename T>
void GetMatrixMarket_CSR_symm( const char* filename,
                               std::vector<T> &data,
                               std::vector<int> &rowPntr,
                               std::vector<int> &icol,
                               int &non_zero,
                               int &row, int &col )
{

  int a,b;
  T c;

  std::ifstream fin(filename);
  while (fin.peek() == '%') fin.ignore(2048, '\n');

  fin>>row>>col>>non_zero;

  std::cerr<<row<<" "<<col<<" "<<non_zero<<std::endl;

  std::vector<std::map<int,double> > mm(row);

  while (fin>>a>>b>>c){
    mm[a-1][b-1] = c;
 //   mm[b-1][a-1] = c;
  }

  // std::cerr<<"mm loaded"<<std::endl;

  rowPntr.push_back(0);

  for(int i=0; i<row; ++i)
  {
    int last = *(rowPntr.rbegin());
    rowPntr.push_back(last+mm[i].size());
    for (std::map<int,double>::iterator it=mm[i].begin(); it!=mm[i].end(); ++it)
      {
        data.push_back(it->second);
        icol.push_back(it->first);
      }
  } 

  non_zero=data.size();

}
 
