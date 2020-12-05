const realFileBtn = document.getElementById("file_");
const customBtn = document.getElementById("button1");
const customTxt = document.getElementById("custom-text");

customBtn.addEventListener("click", function() {
  realFileBtn.click();
});

realFileBtn.addEventListener("change", function() {
  if (realFileBtn.value) {
    customTxt.innerHTML = realFileBtn.value.match(
      /[\/\\]([\w\d\s\.\-\(\)]+)$/
    )[1];
  } else {
    customTxt.innerHTML = "";
  }
});

var map = null;
    var markerArray = [];


    function submit(){

    var val = $("#id_one").val();
    var xhr = new XMLHttpRequest();
    xhr.open('POST', "http://35.239.255.25:5000", true);
    data = new FormData();
    data.append("type", val);
    if(val != "cancer")
      {console.log("Here");
       console.log(data);
    data.append("img", $("#file_").prop('files')[0]);}
    xhr.onreadystatechange = function(){

      var obj = JSON.parse(this.responseText)[0];
      if(!obj["empty"])
      {
        document.getElementById("demo").innerHTML = "Analysis Report";
        //clear();
        $("#result").html("<b>"+obj["pred_val"]+"</b>")

      }
    };
    xhr.send(data);
  }


  function changeText(button, text, textToChangeBackTo) {
  buttonId = document.getElementById(button);
  buttonId.textContent = text;
  setTimeout(function() { back(buttonId, textToChangeBackTo); }, 10000);
  function back(button, textToChangeBackTo){ button.textContent = textToChangeBackTo; }
}
