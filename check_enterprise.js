// 一行指令检查是否为 Enterprise 版本
!!(typeof grecaptcha !== 'undefined' && grecaptcha.enterprise) ? console.log("✅ Enterprise 版本") : console.log("❌ 普通版本");
