document.addEventListener("DOMContentLoaded", function () {
  const bars = document.querySelectorAll(".progress");
  bars.forEach(function (bar) {
    const perc = Number(bar.getAttribute("data-perc")) || 0;
    const fill = bar.querySelector(".progress-fill");
    setTimeout(function () {
      fill.style.width = perc + "%";
    }, 100);
  });
});

// Modal işlemleri
function openModal(drug) {
  const modal = document.getElementById("infoModal");
  modal.style.display = "flex";

  const info = {
    alcohol: {
      title: "Alkol",
      description:
        "Alkol, etanol içeren içecekleri kapsar ve merkezi sinir sistemi üzerinde yatıştırıcı etkisi vardır. Fiziksel ve psikolojik bağımlılık yapabilir.",
      found: "Bira, şarap, viski, votka, likör ve bazı hazır kokteyller",
      tips: "Tüketimi sınırlandırın. Sosyal ortamlarda alternatif içecekler tercih edin. Alkol kullanımı konusunda farkındalık eğitimleri ve destek gruplarına katılın. Gerekirse profesyonel danışmanlık alın.",
    },
    amphet: {
      title: "Amfetamin",
      description:
        "Amfetaminler uyarıcı etkisi olan sentetik maddelerdir, dikkat ve enerji hissi artırır. Bağımlılık riski yüksektir.",
      found: "Reçetesiz stimülanlar, sahte enerji hapları, yasadışı tabletler",
      tips: "Kullanımını tamamen bırakın. Enerji ihtiyacı için düzenli uyku ve egzersiz programları oluşturun. Psikolojik destek ve bağımlılık rehabilitasyonu alın.",
    },
    amyl: {
      title: "Amil Nitrat",
      description:
        "Kısa süreli euforia ve rahatlama sağlayan inhalasyon maddesidir. Damar genişletici etkisi vardır, ama ciddi riskler içerir.",
      found: "Ticari olarak 'poppers' adıyla satılan inhaler ürünler",
      tips: "Kullanımı bırakın. Vücudunuzda kalp ve tansiyon risklerini artırır. Bağımlılıkla mücadele için danışmanlık ve bilinçlendirme programlarına katılın.",
    },
    benzos: {
      title: "Benzodiazepin",
      description:
        "Anksiyete ve uykusuzluk tedavisinde kullanılan reçeteli ilaçlardır. Fiziksel ve psikolojik bağımlılık yapabilir.",
      found: "Reçeteli haplar (diazepam, lorazepam, alprazolam)",
      tips: "Doktor kontrolü olmadan kullanmayın. Ani kesilmemeli, dozaj ayarlamaları profesyonel rehberle yapılmalı. Alternatif terapi ve destek programlarını değerlendirin.",
    },
    caff: {
      title: "Kafein",
      description:
        "Kafein uyarıcı bir maddedir, yorgunluk hissini azaltır ve dikkat artırır. Aşırı kullanım bağımlılık ve çarpıntı yapabilir.",
      found: "Kahve, çay, enerji içecekleri, çikolata",
      tips: "Günlük 400 mg’ı aşmayın. Su tüketimini artırın. Yavaş azaltma yöntemiyle tüketimi dengeleyin. Kafeinli ürünleri sabah ve öğle saatlerine kaydırın.",
    },
    cannabis: {
      title: "Kenevir (Cannabis)",
      description:
        "Kenevir bitkisinden elde edilir, psikoaktif etki yapan THC içerir. Hafıza, dikkat ve motivasyon üzerinde etkisi vardır.",
      found: "Esrar sigarası, kenevir çayı, reçeteli tıbbi ürünler",
      tips: "Kullanımı azaltın veya bırakın. Günlük yaşam aktivitelerini olumsuz etkiler. Psikolojik destek ve bağımlılık danışmanlığı alın.",
    },
    choc: {
      title: "Çikolata",
      description:
        "Kakao ve şeker içeren tatlı bir üründür. Düşük düzeyde bağımlılık yaratabilir, aşırı tüketim kilo ve diş sorununa yol açar.",
      found: "Bitter, sütlü veya dolgulu çikolatalar, şekerlemeler",
      tips: "Tüketimi sınırlayın. Şeker yerine meyve veya bitter çikolata tercih edin. Sağlıklı alternatifler planlayın.",
    },
    coke: {
      title: "Kokain",
      description:
        "Kokain güçlü bir uyarıcıdır. Merkezi sinir sistemi üzerinde hızlı bağımlılık yapar, kalp ve psikolojik sorun riskini artırır.",
      found: "Beyaz toz formu, bazı illegal tabletler ve karışımlar",
      tips: "Hemen kullanımı bırakın. Rehabilitasyon ve profesyonel destek şarttır. Sosyal çevrenizi ve tetikleyici ortamları değiştirin.",
    },
    crack: {
      title: "Crack Kokain",
      description:
        "Kokainin kristal formudur, hızlı ve güçlü bağımlılık yapar. Solunum ve kalp üzerinde ciddi riskleri vardır.",
      found: "Kristal tablet veya inhalasyon formu",
      tips: "Acil profesyonel yardım alın. Psikolojik destek ve rehabilitasyon programlarına katılın. Tetikleyici ortamlardan uzak durun.",
    },
    ecstasy: {
      title: "Ecstasy (MDMA)",
      description:
        "Parti ve sosyal etkinliklerde kullanılan uyarıcı ve halüsinatif etkili bir maddedir. Bağımlılık ve ruhsal sorun riski vardır.",
      found: "Tablet veya kapsül formunda partiler ve festivaller",
      tips: "Kullanımı bırakın. Sosyal etkinliklerde tetikleyici ortamlardan uzak durun. Psikolojik danışmanlık ve destek gruplarına katılın.",
    },
    heroin: {
      title: "Heroin",
      description:
        "Afyon türevi opiat, hızlı bağımlılık yapar. Fiziksel, psikolojik ve sosyal çöküşe yol açar.",
      found: "Toz veya tablet formu, bazı illegal karışımlar",
      tips: "Derhal profesyonel yardım alın. Rehabilitasyon şarttır. Sosyal destek ve takip programları kritik önemdedir.",
    },
    ketamine: {
      title: "Ketamin",
      description:
        "Dissosiyatif anestezik, kısa süreli halüsinasyon ve uyum bozukluğu yaratabilir. Bağımlılık potansiyeli vardır.",
      found: "Tablet, sıvı veya toz formu",
      tips: "Kullanımı bırakın. Psikolojik destek ve bilinçlendirme programlarına katılın. Fiziksel ve ruhsal etkiler için doktor takibi şarttır.",
    },
    legalh: {
      title: "Legal Highs",
      description:
        "Sentetik psikoaktif maddeler, yasadışı veya yasal boşlukları kullanır. Bağımlılık ve ciddi sağlık riskleri taşır.",
      found: "Sentetik türevler, internetten veya sokak satışı",
      tips: "Kesinlikle kullanmayın. Bilinçlendirme ve eğitim programlarına katılın. Psikolojik danışmanlık alın.",
    },
    lsd: {
      title: "LSD",
      description:
        "Güçlü halüsinatif etkisi olan bir maddedir. Fiziksel bağımlılık nadirdir ama psikolojik etkiler ciddi olabilir.",
      found: "Kağıt, jel veya sıvı formu",
      tips: "Kullanmayın. Psikolojik destek ve eğitimle bilinçlendirme önemlidir. Tetikleyici ortamlardan uzak durun.",
    },
    meth: {
      title: "Metamfetamin",
      description:
        "Son derece bağımlılık yapan güçlü bir uyarıcıdır. Psikolojik çöküş ve fiziksel bozulma yapar.",
      found: "Kristal veya tablet formu, illegal sentetik ürünler",
      tips: "Hemen kullanımı bırakın. Profesyonel rehabilitasyon ve psikolojik destek alın. Sağlıklı sosyal çevre ve aktiviteler kritik.",
    },
    mushrooms: {
      title: "Sihirli Mantarlar",
      description:
        "Psikoaktif mantarlarda psilosibin bulunur. Halüsinasyon ve algı değişikliklerine yol açabilir.",
      found: "Doğal mantarlar veya kapsül takviyeler",
      tips: "Kullanımı sınırlayın veya bırakın. Güvenli rehberlik ve psikolojik danışmanlık alın. Tetikleyici ortamlardan uzak durun.",
    },
    nicotine: {
      title: "Nikotin",
      description:
        "Sigara, puro ve elektronik sigarada bulunan bağımlılık yapan uyarıcıdır. Kalp ve akciğer hastalıklarına yol açar.",
      found: "Sigara, elektronik sigara, tütün ürünleri",
      tips: "Bırakma programlarına katılın. Nikotin sakızı veya bantları ile kontrollü bırakma. Psikolojik destek ve tetikleyici ortamdan uzak durmak önemlidir.",
    },
    semer: {
      title: "Sentetik Maddeler",
      description:
        "Çeşitli kimyasal bileşimlerden oluşan sentetik maddeler yüksek bağımlılık ve sağlık riski taşır.",
      found: "Sokak satışı, online ürünler",
      tips: "Kesinlikle kullanmayın. Profesyonel destek ve rehabilitasyon şart. Sosyal çevreyi değiştirin ve bilinçlendirme programlarına katılın.",
    },
    vsa: {
      title: "Uçucu Maddeler (VSA)",
      description:
        "Boya tiner, yapıştırıcı gibi uçucu ürünlerin inhalasyonu ciddi beyin ve akciğer hasarına yol açabilir.",
      found: "Evlerde, atölyelerde, depo ürünleri",
      tips: "Kesinlikle kullanmayın. Eğitim ve destek programlarına katılın.",
    },
  };
  // Varsayılan değerler
  const data = info[drug] || {
    title: drug,
    description: "Bu madde hakkında detaylı bilgi yok.",
    found: "-",
    tips: "-",
  };

  document.getElementById("modalTitle").textContent = data.title;
  document.getElementById("modalDescription").textContent = data.description;
  document.getElementById("modalFound").textContent =
    "Bulunduğu yerler: " + data.found;
  document.getElementById("modalTips").textContent = "Öneriler: " + data.tips;
}

function closeModal() {
  document.getElementById("infoModal").style.display = "none";
}

// Modal dışına tıklayınca kapatma
window.onclick = function (event) {
  const modal = document.getElementById("infoModal");
  if (event.target === modal) {
    modal.style.display = "none";
  }
};
