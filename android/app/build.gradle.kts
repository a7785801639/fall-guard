plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// Номер сборки и ключ подписи приходят из CI (секреты GitHub) или из ~/.gradle/gradle.properties.
// Без ключа релиз подписывается отладочным — так проект собирается у любого.
fun prop(name: String): String? = (project.findProperty(name) as String?) ?: System.getenv(name)

val buildNumber = prop("FG_VERSION_CODE")?.toIntOrNull() ?: 2
val keystorePath = prop("FG_KEYSTORE_FILE")

android {
    namespace = "app.fallguard"
    compileSdk = 35

    defaultConfig {
        applicationId = "app.fallguard"
        minSdk = 27
        targetSdk = 35
        versionCode = buildNumber
        versionName = "1.$buildNumber"
    }

    signingConfigs {
        if (keystorePath != null && file(keystorePath).exists()) {
            create("release") {
                storeFile = file(keystorePath)
                storePassword = prop("FG_KEYSTORE_PASSWORD")
                keyAlias = prop("FG_KEY_ALIAS") ?: "fallguard"
                keyPassword = prop("FG_KEY_PASSWORD") ?: prop("FG_KEYSTORE_PASSWORD")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.findByName("release") ?: signingConfigs.getByName("debug")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    lint {
        abortOnError = false
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    // Сканер QR из Google Play services: без разрешения камеры, свой экран сканирования
    implementation("com.google.android.gms:play-services-code-scanner:16.1.0")
    testImplementation("junit:junit:4.13.2")
}
