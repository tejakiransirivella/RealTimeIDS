#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<queue>
#include<map>
#include<omp.h> 
#include<cmath>
#include<set>


using namespace std;

vector<string> split(const string& str, char delimiter){
  vector<string> internal;
  string temp="";
  for(int i=0; i<str.size(); i++){
    if(str[i]==delimiter){
      internal.push_back(temp);
      temp="";
    }else{
      temp+=str[i];
    }
  }
  internal.push_back(temp);
  return internal;
}

bool is_correct(int index,vector<vector<double>> data,vector<string> labels,int k){
  vector<double> dist;
  auto comparator=[&](int a, int b){
    return dist[a]<dist[b];
  };
  priority_queue<int, vector<int>, decltype(comparator)> pq(comparator);
  for(int i=0; i<data.size(); i++){
    if(i!=index){
      double d=0;
      for(int j=0; j<data[i].size(); j++){
        d+=(data[i][j]-data[index][j])*(data[i][j]-data[index][j]);
      }
      dist.push_back(d);
      pq.push(i);
    }
  }
  string label=labels[index];
  map<string, int> freq;
  freq[label]=0;
  for(int i=0; i<k; i++){
    int idx=pq.top();
    pq.pop();
    if (freq.find(labels[idx])==freq.end())
      freq[labels[idx]]=1;
    else
      freq[labels[idx]]++;
  }
  string max_label;
  int max_freq=0;
  for(auto it=freq.begin(); it!=freq.end(); it++){
    if(it->second>max_freq){
      max_freq=it->second;
      max_label=it->first;
    }
  }
  if(freq[label]<max_freq){
    return false;
  }
  return true;
}

void normalize(vector<vector<double>> & data){

  int columns=data[0].size();
  vector<double> max_val(columns, -1e9);
  vector<double> min_val(columns, 1e9);
  for(int i=0; i<data.size(); i++){
    for(int j=0; j<data[i].size(); j++){
      max_val[j]=max(max_val[j], data[i][j]);
      min_val[j]=min(min_val[j], data[i][j]);
    }
  }
  for(int i=0; i<data.size(); i++){
    for(int j=0; j<data[i].size(); j++){
      data[i][j]=(data[i][j]-min_val[j])/(max_val[j]-min_val[j]);
    }
  }

}


void knn(vector<vector<double>> data, vector<string> labels){
  
  int k=50;
  int count=0;
  int size=1000;
  #pragma omp parallel for reduction(+:count)
  for(int i=0;i<size;i++){
    if(is_correct(i,data,labels,k)){
      count++;
    }
  }
  cout<<"Accuracy: "<<(double)count/size<<endl;
}


int main(){
  ifstream file("data/train.csv");
  string line;
  vector<string> header;
  getline(file, line);
  header=split(line, ',');
  cout<<header.size()<<endl;
  vector<vector<double>> data;
  vector<string> labels;
  while(getline(file, line)){
    vector<string> row=split(line, ',');
    vector<double> drow;
    for(int i=0; i<row.size()-1; i++){
      try{
        drow.push_back(stod(row[i]));
      }catch(exception e){
        cout<<line<<endl;
        exit(1);
      }

    }
    labels.push_back(row[row.size()-1]);
    data.push_back(drow);
  }
  normalize(data);
  knn(data, labels);

  return 0;
}