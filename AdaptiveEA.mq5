//+------------------------------------------------------------------+
//|                                                  AdaptiveEA.mq5  |
//|                        Example Expert Advisor using ML signals   |
//+------------------------------------------------------------------+
#property copyright "Yegor Sokolov"
#property version   "1.00"
#property strict


#include <Trade/Trade.mqh>
CTrade trade;

#import "libzmq.dll"
int  zmq_ctx_new();
int  zmq_ctx_term(int ctx);
int  zmq_socket(int ctx,int type);
int  zmq_connect(int sock,string addr);
int  zmq_setsockopt(int sock,int option,uchar &opt[],int len);
int  zmq_recv(int sock,uchar &buf[],int len,int flags);
int  zmq_close(int sock);
#define ZMQ_SUB       2
#define ZMQ_SUBSCRIBE 6
#define ZMQ_DONTWAIT  1
#import

string LogFile = "logs/ea_trades.csv";

input string ZmqAddress = "tcp://localhost:5555"; // address of the signal queue
input double RiskPerTrade = 0.01;
input int TrailingStopPips = 20;
input double MaxDailyLoss = 3.0;   // percent
input double MaxDrawdown = 10.0;   // percent
input int    RiskLookbackBars = 50;      // bars used to estimate risk
input bool   UseSharpeSizing   = true;   // scale position size by Sharpe ratio
input double MinRiskFactor     = 0.5;    // lower bound for position multiplier
input double MaxRiskFactor     = 2.0;    // upper bound for position multiplier
input int    VarLookbackBars   = 50;     // bars used for VaR/stress calc
input double MaxVaR            = 5.0;    // max allowed 99% VaR in percent
input double MaxStressLoss     = 15.0;   // max allowed stress loss percent
input double MaxCVaR           = 7.5;    // max allowed expected shortfall
input double VarDecay          = 0.94;   // decay factor for EWMA VaR
input int    ShortVolPeriod    = 10;     // short-term volatility bars
input int    LongVolPeriod     = 50;     // long-term volatility bars
input int    SignalTimeTolerance = 60;   // seconds tolerance for matching signals

int      zmq_ctx       = 0;
int      zmq_sock      = 0;
double   last_prob     = 0.0;
datetime last_sig_time = 0;

double peak_equity = 0.0;
double day_start_equity = 0.0;
datetime last_day = 0;
bool trading_allowed = true;

void WriteLog(string event, double val=0.0)
{
   int fh = FileOpen(LogFile, FILE_READ|FILE_WRITE|FILE_CSV|FILE_TXT|FILE_ANSI);
   if(fh == INVALID_HANDLE)
   {
      fh = FileOpen(LogFile, FILE_WRITE|FILE_CSV|FILE_TXT|FILE_ANSI);
      if(fh == INVALID_HANDLE)
         return;
      FileWrite(fh, "timestamp", "event", "value");
   }
   FileSeek(fh, 0, SEEK_END);
   FileWrite(fh, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), event, val);
   FileClose(fh);
}


bool UpdateSignal()
{
   uchar buf[];
   ArrayResize(buf,256);
   int len = zmq_recv(zmq_sock,buf,ArraySize(buf),ZMQ_DONTWAIT);
   if(len<=0)
      return(false);
   string msg = CharArrayToString(buf,0,len);
   int tp = StringFind(msg,"Timestamp");
   int pp = StringFind(msg,"prob");
   if(tp<0 || pp<0)
      return(false);
   int s = StringFind(msg,"\"",tp+9);
   int e = StringFind(msg,"\"",s+1);
   if(s<0 || e<0)
      return(false);
   string ts = StringSubstr(msg,s+1,e-s-1);
   last_sig_time = StringToTime(ts);
   string rest = StringSubstr(msg,pp+5);
   int end = StringFind(rest,"}");
   if(end>0)
      rest = StringSubstr(rest,0,end);
   last_prob = StrToDouble(rest);
   return(true);
}

double CalculateRiskFactor()
{
   int count=MathMin(RiskLookbackBars,Bars(Symbol(),PERIOD_CURRENT)-1);
   if(count<=1)
      return(1.0);

   double mean=0.0;
   for(int i=0;i<count;i++)
      mean+=(Close[i]-Close[i+1])/Close[i+1];
   mean/=count;

   double var=0.0;
   for(int i=0;i<count;i++)
   {
      double ret=(Close[i]-Close[i+1])/Close[i+1];
      var+=MathPow(ret-mean,2);
   }
   double sd=MathSqrt(var/count);
   if(sd==0.0)
      return(1.0);

   double factor;
   if(UseSharpeSizing)
   {
      double sharpe=mean/sd;
      factor=1.0+sharpe;
   }
   else
   {
      factor=1.0/(1.0+sd*100.0);
   }

   double regime = RegimeMultiplier();

   if(factor<MinRiskFactor)
      factor=MinRiskFactor;
   if(factor>MaxRiskFactor)
      factor=MaxRiskFactor;

   return(factor*regime);
}

double RegimeMultiplier()
{
   int short_n=MathMin(ShortVolPeriod,Bars(Symbol(),PERIOD_CURRENT)-1);
   int long_n=MathMin(LongVolPeriod,Bars(Symbol(),PERIOD_CURRENT)-1);
   if(long_n<=1 || short_n<=1)
      return(1.0);

   double mean_s=0.0, mean_l=0.0;
   for(int i=0;i<short_n;i++)
      mean_s+=(Close[i]-Close[i+1])/Close[i+1];
   mean_s/=short_n;
   for(int i=0;i<long_n;i++)
      mean_l+=(Close[i]-Close[i+1])/Close[i+1];
   mean_l/=long_n;

   double var_s=0.0, var_l=0.0;
   for(int i=0;i<short_n;i++)
   {
      double r=(Close[i]-Close[i+1])/Close[i+1];
      var_s+=MathPow(r-mean_s,2);
   }
   for(int i=0;i<long_n;i++)
   {
      double r=(Close[i]-Close[i+1])/Close[i+1];
      var_l+=MathPow(r-mean_l,2);
   }
   double sd_s=MathSqrt(var_s/short_n);
   double sd_l=MathSqrt(var_l/long_n);
   if(sd_l==0.0)
      return(1.0);
   double ratio=sd_s/sd_l;
   if(ratio>1.5)
      return(0.5);
   if(ratio<0.7)
      return(1.2);
   return(1.0);
}

double CalculateVaR()
{
   int count=MathMin(VarLookbackBars,Bars(Symbol(),PERIOD_CURRENT)-1);
   if(count<=1)
      return(0.0);
   double arr[];
   ArrayResize(arr,count);
   for(int i=0;i<count;i++)
      arr[i]=(Close[i]-Close[i+1])/Close[i+1];
   ArraySort(arr,WHOLE_ARRAY,0,MODE_ASCEND);
   int idx=(int)MathFloor(0.01*count);
   double var=-arr[idx]*100.0;
   return(var);
}

double CalculateCVaR()
{
   int count=MathMin(VarLookbackBars,Bars(Symbol(),PERIOD_CURRENT)-1);
   if(count<=1)
      return(0.0);
   double arr[];
   ArrayResize(arr,count);
   for(int i=0;i<count;i++)
      arr[i]=(Close[i]-Close[i+1])/Close[i+1];
   ArraySort(arr,WHOLE_ARRAY,0,MODE_ASCEND);
   int idx=(int)MathFloor(0.01*count);
   double sum=0.0;
   for(int i=0;i<=idx;i++)
      sum+=arr[i];
   double cvar=-(sum/(idx+1))*100.0;
   return(cvar);
}

double CalculateFilteredVaR()
{
   int count=MathMin(VarLookbackBars,Bars(Symbol(),PERIOD_CURRENT)-1);
   if(count<=1)
      return(0.0);
   double decay=VarDecay;
   double weight=1.0;
   double sum_w=0.0;
   double mean=0.0;
   for(int i=0;i<count;i++)
   {
      double r=(Close[i]-Close[i+1])/Close[i+1];
      mean+=weight*r;
      sum_w+=weight;
      weight*=decay;
   }
   mean/=sum_w;

   weight=1.0;
   sum_w=0.0;
   double var=0.0;
   for(int i=0;i<count;i++)
   {
      double r=(Close[i]-Close[i+1])/Close[i+1];
      var+=weight*MathPow(r-mean,2);
      sum_w+=weight;
      weight*=decay;
   }
   var/=sum_w;
   double sd=MathSqrt(var);
   double var_pct=2.33*sd*100.0;
   return(var_pct);
}

double CalculateStressLoss()
{
   int count=MathMin(VarLookbackBars,Bars(Symbol(),PERIOD_CURRENT)-1);
   if(count<=1)
      return(0.0);
   double mean=0.0;
   for(int i=0;i<count;i++)
      mean+=(Close[i]-Close[i+1])/Close[i+1];
   mean/=count;
   double variance=0.0;
   for(int i=0;i<count;i++)
   {
      double r=(Close[i]-Close[i+1])/Close[i+1];
      variance+=MathPow(r-mean,2);
   }
   double sd=MathSqrt(variance/count);
   return(3.0*sd*100.0);
}

void CloseAllPositions()
{
   while(PositionSelect(Symbol()))
   {
      ulong ticket=PositionGetTicket(0);
      trade.PositionClose(ticket);
      WriteLog("close", ticket);
   }
}

void UpdateRisk()
{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(equity>peak_equity)
      peak_equity=equity;

   datetime today = Date();
   if(today!=last_day)
   {
      day_start_equity=equity;
      last_day=today;
      trading_allowed=true;
   }

   double day_loss_pct=(equity-day_start_equity)/day_start_equity*100.0;
   double drawdown_pct=(equity-peak_equity)/peak_equity*100.0;
   double var_pct=CalculateFilteredVaR();
   double cvar_pct=CalculateCVaR();
   double stress_pct=CalculateStressLoss();
  if(day_loss_pct<=-MaxDailyLoss || drawdown_pct<=-MaxDrawdown)
  {
      CloseAllPositions();
      trading_allowed=false;
      WriteLog("risk_pause", day_loss_pct);
  }
  if(var_pct>MaxVaR || stress_pct>MaxStressLoss || cvar_pct>MaxCVaR)
  {
      CloseAllPositions();
      trading_allowed=false;
      WriteLog("risk_pause", var_pct);
  }
}

int OnInit()
{
   zmq_ctx  = zmq_ctx_new();
   zmq_sock = zmq_socket(zmq_ctx,ZMQ_SUB);
   zmq_connect(zmq_sock,ZmqAddress);
   uchar filter[]; ArrayResize(filter,0);
   zmq_setsockopt(zmq_sock,ZMQ_SUBSCRIBE,filter,0);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   if(zmq_sock!=0) zmq_close(zmq_sock);
   if(zmq_ctx!=0)  zmq_ctx_term(zmq_ctx);
}

void OnTick()
{
   static datetime last_time=0;
   if(TimeCurrent()==last_time)
      return;
   last_time=TimeCurrent();

   UpdateRisk();
   if(!trading_allowed)
      return;

   UpdateSignal();
   if(TimeCurrent()-last_sig_time>SignalTimeTolerance)
      return;
   double prob = last_prob;

   if(PositionSelect(Symbol())==false && prob>0.55)
   {
      double risk_factor=CalculateRiskFactor();
      double volume=NormalizeDouble(AccountBalance()*RiskPerTrade*risk_factor/1000,2);
      trade.Buy(volume,NULL,Ask,0,0);
      WriteLog("open", prob);
   }

   if(PositionSelect(Symbol()))
   {
      ulong ticket=PositionGetTicket(0);
      double stop=PositionGetDouble(POSITION_SL);
      double price=SymbolInfoDouble(Symbol(),SYMBOL_BID);
      double new_stop=price - TrailingStopPips*_Point;
      if(new_stop>stop)
      {
         trade.PositionModify(ticket,new_stop,0);
         WriteLog("trailing_stop", new_stop);
      }
   }
}
