#Analisis Sentimen Terhadap Tweet Masyarakat Indonesia 
#Tentang Covid dan Pemerintah
#Proyek Akhir Praktikum Data Science Plug A
#Atania Harfiani (123180092)
#Fauziyah Ahmad Inayanti (123180097)

library(vroom)
library(here)
library(sentimentr)
library(tidytext)
library(textclean)
library(tibble)
library(tm)

#Load Dataset
rawData=read.csv("file:///D:/Ina/upn/SEMESTER 5/Data Science/proyek/covid.csv",header=T)
#Mengubah Data (Tabel Tweet) Menjadi Vector
tweets=rawData$tweet
tweets.text=Corpus(VectorSource(tweets))

##Cleaning data
removeURL <- function(x) gsub("http[^[:space:]]*", "", x)
clean <- tm_map(tweets.text, removeURL)

removeNL <- function(y) gsub("\n", " ", y)
clean <- tm_map(clean, removeNL)

removepipe <- function(z) gsub("<[^>]+>", "", z)
clean <- tm_map(clean, removepipe)

remove.mention <- function(z) gsub("@\\S+", "", z)
clean <- tm_map(clean, remove.mention)

remove.hashtag <- function(z) gsub("#\\S+", "", z)
clean <- tm_map(clean, remove.hashtag)

removeamp <- function(y) gsub("&amp;", "", y)
clean <- tm_map(clean, removeamp)

removetitik3 <- function(y) gsub("[[:punct:]]", "", y)
clean <- tm_map(clean, removetitik3)

remove.all <- function(xy) gsub("[^[:alpha:][:space:]]*", "", xy)
clean <- tm_map(clean,remove.all)

clean <- tm_map(clean, tolower)

#remove extra whitespace (spasi)
clean <- tm_map(clean, stripWhitespace)

##load stopword-ID
stopwordID <- "D:/Ina/upn/SEMESTER 5/Data Science/proyek/ind/Cleaning-Text-Bahasa-Indonesia-master/ID-Stopwords.txt"
##membaca stopwordID perbaris
cStopwordID<-readLines(stopwordID);

#load slangword (bahasa gaul)
slang <- read.csv("file:///D:/Ina/upn/SEMESTER 5/Data Science/proyek/ind/Cleaning-Text-Bahasa-Indonesia-master/Slangword.csv", header=T)
old_slang <- as.character(slang$old) 
new_slang <- as.character(slang$new)
#load stemming (kata imbuhan)
stemm <- read.csv("file:///D:/Ina/upn/SEMESTER 5/Data Science/proyek/ind/Cleaning-Text-Bahasa-Indonesia-master/Stemming.csv", header=T)
old_stemm <- as.character(stemm$old)
new_stemm <- as.character(stemm$new)
#load lemmatization (pengelompokan infleksi kata)
lemma <- read.csv("file:///D:/Ina/upn/SEMESTER 5/Data Science/proyek/ind/Cleaning-Text-Bahasa-Indonesia-master/Lemmatization.csv", header=T)
old_lemma <- as.character(stemm$old)
new_lemma <- as.character(stemm$new)

stemmword <- function(x) Reduce(function(x,r) gsub(stemm$old[r],stemm$new[r],x,fixed=T),
                                seq_len(nrow(stemm)),x)
clean <- tm_map(clean,stemmword)
slangword <- function(x) Reduce(function(x,r) gsub(slang$old[r],slang$new[r],x,fixed=T),
                                seq_len(nrow(slang)),x)
clean <- tm_map(clean,slangword)
lemmatization <- function(x) Reduce(function(x,r) gsub(lemma$old[r],lemma$new[r],x,fixed=T),
                                    seq_len(nrow(lemma)),x)
clean <- tm_map(clean,lemmatization)

clean <- tm_map(clean, removeWords, cStopwordID)
writeLines(strwrap(clean[[2]]$content, 100))

##Save data
dataframe=data.frame(text=unlist(sapply(clean, `[`)), stringsAsFactors=F)
View(dataframe)
write.csv(dataframe,file = 'D:/Ina/upn/SEMESTER 5/Data Science/proyek/DataBersih.csv')

#SENTIMENT 
library(e1071)
library(caret)
library(syuzhet)
#Membaca file csv yang sudah di cleaning data 
datanya<-read.csv("D:/Ina/upn/SEMESTER 5/Data Science/proyek/DataBersih.csv",stringsAsFactors = FALSE)

#Set variabel column text menjadi char
tweetss <-as.character(datanya$tweet)

#Memanggil NRC sentiment dictionary untuk mengkalkulasi berbagai emosi 
s<-get_nrc_sentiment(tweets)

tweets_combine<-cbind(datanya$tweet,s)
par(mar=rep(3,4))
a<- barplot(colSums(s),col=rainbow(10),ylab='count',main='sentiment analisis')
iki_ba <- a

#WORDCLOUD
#Library untuk membuat wordcloud
library(wordcloud)
#Library untuk penggunaan corpus dalam cleaning data
library(tm)
library(RTextTools)
#Library yang terdapat sebuah algoritma naivebayes
library(e1071)
library(dplyr)
library(caret)

#Membaca data yang sudah dibersihkan 
df<-read.csv("D:/Ina/upn/SEMESTER 5/Data Science/proyek/DataBersih.csv",stringsAsFactors = FALSE)
glimpse(df)

#Set the seed of R's random number generator, which is useful for creating simulations or random objects that can be reproduced.
set.seed(20)
df<-df[sample(nrow(df)),]
df<-df[sample(nrow(df)),]
glimpse(df)

corpus<-Corpus(VectorSource(df$text))
corpus
inspect(corpus[1:10])

#Fungsinya untuk membersihkan data-data yang tidak dibutuhkan 
corpus.clean<-corpus%>%
  tm_map(content_transformer(tolower))%>%
  tm_map(removePunctuation)%>%
  tm_map(removeNumbers)%>%
  tm_map(removeWords,stopwords(kind="en"))%>%
  tm_map(stripWhitespace)
dtm<-DocumentTermMatrix(corpus.clean)

inspect(dtm[1:10,1:20])

df.train<-df[1:50,]
df.test<-df[51:100,]

dtm.train<-dtm[1:50,]
dtm.test<-dtm[51:100,]

corpus.clean.train<-corpus.clean[1:50]
corpus.clean.test<-corpus.clean[51:100]

dim(dtm.train)
fivefreq<-findFreqTerms(dtm.train,5)
length(fivefreq)

dtm.train.nb<-DocumentTermMatrix(corpus.clean.train,control = list(dictionary=fivefreq))

#dim(dtm.train.nb)

dtm.test.nb<-DocumentTermMatrix(corpus.clean.test,control = list(dictionary=fivefreq))

dim(dtm.test.nb)

convert_count <- function(x){
  y<-ifelse(x>0,1,0)
  y<-factor(y,levels=c(0,1),labels=c("no","yes"))
  y
}
trainNB<-apply(dtm.train.nb,2,convert_count)
testNB<-apply(dtm.test.nb,1,convert_count)

#Generate wordcloud
wordcloud(corpus.clean,min.freq = 4,max.words=100,random.order=F,colors=brewer.pal(8,"Dark2"))

#Membuat tampilan shiny
#Library untuk membuat shiny
library(shiny)
#Script UI
ui <- fluidPage(
  
  # Application title
  titlePanel("Analisis Sentimen Terhadap Tweet Masyarakat Indonesia Tentang Covid dan Pemerintah"),
  mainPanel(
    #textOutput("selected_var"),
    
    #plotOutput("asPlot"),
    tabsetPanel(type = "tabs",
                tabPanel("Data Asli", DT::dataTableOutput('tbl')),
                tabPanel("Data Bersih", DT::dataTableOutput('tbl2')),
                tabPanel("Scatterplot", plotOutput("asPlot")),
                tabPanel("Wordcloud", plotOutput("wordcl"))
    )
  )
)

#Script server
#Membuat fungsi `input`, `output`, dan `session`
#Di dalam badan fungsi tersebut berisi seluruh kode pemrosesan data
#Membuat `input` dan menampilkan hasil pada `output`

server <- function(input, output, session) {
  as_data <- reactive({
    
    input$Update
    isolate({
      withProgress({
        setProgress(message = "Processing analisis...")
        as_file <- input$as
        if(!is.null(as_file)){
          as_text <- readLines(as_file$datapath)
        }
        else
        {
          as_text <- "A Barplot is an immage made of words that..."
        }
        
      })
    })
  })
  
  barplot_rep <- repeatable(barplot)
  
  #Menampilkan data asli
  output$tbl = DT::renderDataTable({
    DT::datatable(rawData, options = list(lengthchange = FALSE))
  })
  #Menampilkan data bersih
  output$tbl2 = DT::renderDataTable({
    DT::datatable(datanya, options = list(lengthchange = FALSE))
  })
  #Menampilkan grafik sentimen
  output$asPlot <- renderPlot({ withProgress({
    setProgress(message = "Creating barplot...")
    barplot(colSums(s),col = rainbow(10),ylab = 'count',main = 'Sentiment Analysis')
  })
  })
  #Menampilkan wordcloud
  output$wordcl <- renderPlot({
    wordcloud(corpus.clean,min.freq = 4,max.words=100,random.order=F,colors=brewer.pal(8,"Dark2"))
  })
  
}

#Menjalankan shiny
shinyApp(ui = ui, server = server, options = list(height = "600px"))