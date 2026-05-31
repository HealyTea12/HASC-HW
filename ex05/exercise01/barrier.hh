#ifndef HASC_BARRIER_HH
#define HASC_BARRIER_HH

#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>

// barrier as a class
// it implements two version: with mutexes and without mutexes
class Barrier
{

  int P;                                   // number of threads in barrier
  int count;                               // count number of threads that arrived at the barrier
  std::vector<int> flag;                   // flag indicating waiting thread
  std::mutex mx;                           // mutex for use with the cvs
  std::vector<std::condition_variable> cv; // for waiting
  std::atomic<int> acounter, bcounter;     // two counters for mutex-free version
  std::vector<int> direction;              // store counting direction in mutex-free version
public:
  // set up barrier for given number of threads
  Barrier(int P_) : P(P_), count(0), flag(P_, 0), cv(P_), direction(P, 0)
  {
    acounter.store(0);
    bcounter.store(0);
  }

  // get number of threads
  int nthreads()
  {
    return P;
  }

  // mutex-based version
  void wait(int i)
  {
    // sequential case
    if (P == 1)
      return;

    std::unique_lock<std::mutex> ul{mx};
    count += 1; // one more
    if (count < P)
    {
      // wait on my cv until all have arrived
      flag[i] = 1; // indicate I am waiting
      cv[i].wait(ul, [i, this]
                 { return this->flag[i] == 0; }); // wait
    }
    else
    {
      // I am the last one, lets wake them up
      count = 0; // reset counter for next turn
      for (int j = 0; j < P; j++)
        if (flag[j] == 1)
        {
          flag[j] = 0;        // the event
          cv[j].notify_one(); // wake up
        }
    }
  }
};

class CounterBarrier
{
  int P;                                   // number of threads in barrier
  int count;                               // count number of threads that arrived at the barrier
  std::vector<int> flag;                   // flag indicating waiting thread
  std::mutex mx;                           // mutex for use with the cvs
  std::vector<std::condition_variable> cv; // for waiting
  std::atomic<int> acounter, bcounter;     // two counters for mutex-free version
  std::vector<int> direction;              // store counting direction in mutex-free version
public:
  CounterBarrier(int P_) : P(P_), count(0), flag(P_, 0), cv(P_), direction(P, 0)
  {
    acounter.store(0);
    bcounter.store(0);
  }

  // get number of threads
  int nthreads()
  {
    return P;
  }
  void wait(int i)
  {
    if (direction[i] == 0)
    {
      acounter++;
      while (acounter.load() < P)
        ;
      bcounter++;
      while (bcounter.load() < P)
        ;
    }
    else
    {
      acounter--;
      while (acounter.load() > 0)
        ;
      bcounter--;
      while (bcounter.load() > 0)
        ;
    }
    direction[i] = 1 - direction[i]; // reverse direction in next round
  }
};

class TreeBarrier
{
public:
  TreeBarrier(int p) : m_p(p)
  {
    m_binTree = std::vector<Node>(2 * p - 1);
    for (size_t i{0}; i < m_binTree.size(); i++)
    {
      m_binTree[i].count.store(0);
      m_binTree[i].sense.store(false);
    }
    m_sense = std::vector<std::atomic<bool>>(p);
    for (size_t i{0}; i < m_sense.size(); i++)
    {
      m_sense[i].store(true);
    }
  };

  void wait(int i)
  {
    bool localSense = m_sense[i].load();
    size_t leafIndex = static_cast<size_t>(m_p - 1 + i);
    size_t parent = getParent(leafIndex);
    visit(parent, localSense);
    m_sense[i].store(!localSense);
  }

private:
  void visit(size_t nodeIndex, bool sense)
  {
    Node &node = m_binTree[nodeIndex];
    int count = node.count.fetch_add(1);
    if (count == 1) // last one to visit
    {
      if (getParent(nodeIndex) != nodeIndex)
        visit(getParent(nodeIndex), sense);
      node.count.store(0);
      node.sense.store(sense);
    }
    else
    {
      while (node.sense.load() != sense)
        ;
    }
  }
  size_t getParent(size_t i)
  {
    if (i == 0)
      return 0;
    return (i - 1) / 2;
  }

private:
  struct Node
  {
    std::atomic<int> count;
    std::atomic<bool> sense;
  };
  std::vector<Node> m_binTree;
  std::vector<std::atomic<bool>> m_sense;
  int m_p;
};

#endif
